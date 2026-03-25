#!/usr/bin/env python3
"""
Biomechanics Trial Viewer
Interactive viewer for IMU sensor data and joint moment/angle labels.
Browse trials, visualize all signals, and flag problematic ones.

Keyboard shortcuts (when not typing in an input):
    Left/Right arrows  -  Navigate between trials
    F                  -  Toggle flag on current trial
"""

import os
import sys
import json
import re
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


# ── Trial Discovery ──────────────────────────────────────────────────────────

_SUBJECT_RE = re.compile(r"^S\d{3}$")
_TRIAL_RE = re.compile(r"^trial_\d+$")


def _resolve_h5_pair(root_path: str) -> tuple[str | None, str | None]:
    """Resolve (filesystem subject root, h5 bundle dir) for paired layouts.

    Supports `.../<Name>/Sxxx/...` next to `.../<Name>_h5/Sxxx.h5` (same parent),
    e.g. Camargo/Camargo_h5, Scherpereel/Scherpereel_h5, Molinaro_Scherpereel/..., MetaMobility.

    Returns (fs_root, h5_root). If only the *_h5 tree exists, fs_root is None and
    h5_root points at that directory (h5-only discovery).
    """
    root_path = os.path.abspath(root_path)
    base = os.path.basename(os.path.normpath(root_path))
    parent = os.path.dirname(os.path.normpath(root_path))

    pairs = (
        ("Camargo", "Camargo_h5"),
        ("MetaMobility", "MetaMobility_h5"),
        ("Scherpereel", "Scherpereel_h5"),
        ("Molinaro_Scherpereel", "Molinaro_Scherpereel_h5"),
    )
    for fs_name, h5_name in pairs:
        if base == fs_name:
            h5_root = os.path.join(parent, h5_name)
            if os.path.isdir(h5_root):
                return root_path, h5_root
        elif base == h5_name:
            h5_root = root_path
            fs_root = os.path.join(parent, fs_name)
            if os.path.isdir(fs_root):
                return fs_root, h5_root
            return None, h5_root
    return None, None


def _memo_flat_h5_root(root_path: str) -> str | None:
    """`Processed/MeMo` stores `Sxxx.h5` files in the dataset folder (no MeMo_h5 sibling, no per-subject dirs)."""
    root_path = os.path.abspath(root_path)
    if os.path.basename(os.path.normpath(root_path)) != "MeMo":
        return None
    if not os.path.isdir(root_path):
        return None
    try:
        names = os.listdir(root_path)
    except OSError:
        return None
    if not any(n.endswith(".h5") and _SUBJECT_RE.match(n[:-3]) for n in names):
        return None
    return root_path


def discover_trials(root_path):
    """Discover trials for supported dataset layouts.

    Primary support:
    - Paired filesystem + HDF5: `.../<Name>/Sxxx/<condition>/trial_YY/...` with sibling
      `.../<Name>_h5/Sxxx.h5` (angles + moments in h5). Same pattern for Camargo, Scherpereel,
      Molinaro_Scherpereel, MetaMobility.
    - HDF5-only: open `.../<Name>_h5` when the filesystem tree is absent, or `.../MeMo` (flat Sxxx.h5).
    """
    trials = []
    root_path = os.path.abspath(root_path)

    fs_root, h5_root = _resolve_h5_pair(root_path)
    if h5_root is None:
        memo = _memo_flat_h5_root(root_path)
        if memo is not None:
            h5_root = memo

    # 1) Discover trials from the filesystem tree; load arrays from sibling *_h5/*.h5.
    if fs_root and h5_root and os.path.isdir(fs_root):
        subjects = [d for d in os.listdir(fs_root) if _SUBJECT_RE.match(d)]
        if subjects:
            for subj in sorted(subjects):
                subj_dir = os.path.join(fs_root, subj)
                for condition in sorted(os.listdir(subj_dir)):
                    condition_dir = os.path.join(subj_dir, condition)
                    if not os.path.isdir(condition_dir):
                        continue
                    if not os.path.isfile(os.path.join(condition_dir, "condition_meta.json")):
                        continue
                    for trial in sorted(os.listdir(condition_dir)):
                        if not _TRIAL_RE.match(trial):
                            continue
                        h5_path = os.path.join(h5_root, f"{subj}.h5")
                        if not os.path.isfile(h5_path):
                            continue
                        trials.append({
                            "kind": "camargo_h5",
                            "h5_path": h5_path,
                            "condition": condition,
                            "trial": trial,
                            "label": f"{subj}/{condition}/{trial}",
                        })
            if trials:
                trials.sort(key=lambda t: t["label"])
                return trials

    # 2) HDF5-only, or explicit *_h5 root: discover by inspecting h5 keys.
    if h5_root and os.path.abspath(root_path) == os.path.abspath(h5_root):
        if h5py is None:
            return []
        for fname in sorted(os.listdir(h5_root)):
            if not fname.endswith(".h5"):
                continue
            subj = fname[:-3]
            h5_path = os.path.join(h5_root, fname)
            with h5py.File(h5_path, "r") as f:
                for condition in sorted(f.keys()):
                    for trial in sorted(f[condition].keys()):
                        if not _TRIAL_RE.match(trial):
                            continue
                        trials.append({
                            "kind": "camargo_h5",
                            "h5_path": h5_path,
                            "condition": condition,
                            "trial": trial,
                            "label": f"{subj}/{condition}/{trial}",
                        })
        trials.sort(key=lambda t: t["label"])
        return trials

    # 3) Fallback to the older CSV layout (Input/imu_data.csv, Label/joint_*.csv).
    for dirpath, _, _ in os.walk(root_path):
        if os.path.isfile(os.path.join(dirpath, "Input", "imu_data.csv")):
            trials.append({
                "path": dirpath,
                "label": os.path.relpath(dirpath, root_path),
            })
    trials.sort(key=lambda t: t["label"])
    return trials


# ── Data Loading (with single-trial cache) ───────────────────────────────────

_cache = {"key": None, "data": None}


def _read_h5_columns(dataset) -> list[str]:
    # Stored as JSON string in a `columns` attribute, e.g. '["time", "..."]'
    return json.loads(dataset.attrs["columns"])


def get_trial_data(trial_info):
    cache_key = None
    if isinstance(trial_info, dict):
        kind = trial_info.get("kind")
        if kind == "camargo_h5":
            cache_key = (kind, trial_info["h5_path"], trial_info["condition"], trial_info["trial"])
        elif "path" in trial_info:
            cache_key = ("csv", trial_info["path"])
    else:
        cache_key = ("csv", trial_info)

    if _cache["key"] == cache_key:
        return _cache["data"]

    data = {}

    if isinstance(trial_info, dict) and trial_info.get("kind") == "camargo_h5":
        if h5py is None:
            raise RuntimeError("Missing dependency `h5py`. Install `trial-viewer` with h5py.")

        h5_path = trial_info["h5_path"]
        condition = trial_info["condition"]
        trial_name = trial_info["trial"]

        with h5py.File(h5_path, "r") as f:
            trial_group = f[condition][trial_name]

            # IMU: combine per-sensor datasets into one dataframe.
            imu_group = trial_group["imu"]
            time_vec = None
            imu_data = {}
            for sensor in sorted(imu_group.keys()):
                ds = imu_group[sensor]
                cols = _read_h5_columns(ds)
                arr = ds[...]

                if "time" in cols:
                    t = arr[:, cols.index("time")]
                else:
                    t = arr[:, 0]

                if time_vec is None:
                    time_vec = t

                for j, col in enumerate(cols):
                    if col == "time":
                        continue
                    imu_data[col] = arr[:, j]

            if time_vec is None:
                data["imu"] = pd.DataFrame()
            else:
                imu_df = pd.DataFrame(imu_data)
                imu_df.insert(0, "time", time_vec)
                data["imu"] = imu_df

            # IK -> angles
            ik_group = trial_group["ik"]
            ik_ds = next(iter(ik_group.values()))
            ik_cols = _read_h5_columns(ik_ds)
            ik_arr = ik_ds[...]
            data["angle"] = pd.DataFrame(ik_arr, columns=ik_cols)

            # ID -> moments
            id_group = trial_group["id"]
            id_ds = next(iter(id_group.values()))
            id_cols = _read_h5_columns(id_ds)
            id_arr = id_ds[...]
            data["moment"] = pd.DataFrame(id_arr, columns=id_cols)

        _cache["key"] = cache_key
        _cache["data"] = data
        return data

    # CSV fallback
    if isinstance(trial_info, dict):
        trial_path = trial_info["path"]
    else:
        trial_path = trial_info

    files = {
        "imu": os.path.join(trial_path, "Input", "imu_data.csv"),
        "moment": os.path.join(trial_path, "Label", "joint_moment.csv"),
        "angle": os.path.join(trial_path, "Label", "joint_angle.csv"),
    }
    for key, fpath in files.items():
        if os.path.isfile(fpath):
            data[key] = pd.read_csv(fpath)

    _cache["key"] = cache_key
    _cache["data"] = data
    return data


# ── Plot Builders ─────────────────────────────────────────────────────────────

AXIS_COLORS = {"x": "#e74c3c", "y": "#27ae60", "z": "#2980b9"}
SIDE_STYLES = {
    "l": {"color": "#e74c3c", "label": "Left"},
    "r": {"color": "#2980b9", "label": "Right"},
}


def _pretty(name):
    return name.replace("_", " ").title()


def build_imu_figure(df, sensor_type):
    """Build a multi-row figure for accelerometer or gyroscope data.
    Sensors are auto-detected from column names containing _accel_ or _gyro_."""
    axis_order = {"x": 0, "y": 1, "z": 2}

    sensors = {}

    # Camargo_h5 naming:
    # - accelerometer: `<sensor>_acc_<x|y|z>` (cols like `pelvis_acc_x`)
    # - gyroscope:     `<sensor>_gyr_<x|y|z>` (cols like `pelvis_gyr_y`)
    if sensor_type == "accel":
        camargo_re = re.compile(r"(?P<sensor>.+)_acc_(?P<axis>[xyz])$")
        for col in df.columns:
            m = camargo_re.match(col)
            if not m:
                continue
            sensors.setdefault(m.group("sensor"), []).append((m.group("axis"), col))
    elif sensor_type == "gyro":
        camargo_re = re.compile(r"(?P<sensor>.+)_gyr_(?P<axis>[xyz])$")
        for col in df.columns:
            m = camargo_re.match(col)
            if not m:
                continue
            sensors.setdefault(m.group("sensor"), []).append((m.group("axis"), col))

    # Older CSV naming (keep as fallback)
    if not sensors:
        key = f"_{sensor_type}_"
        relevant = [c for c in df.columns if key in c]
        for col in relevant:
            parts = col.split(key)
            if len(parts) != 2:
                continue
            sensor, axis = parts
            sensors.setdefault(sensor, []).append((axis, col))

    if not sensors:
        return _empty_figure("No data available")

    names = sorted(sensors.keys())
    fig = make_subplots(
        rows=len(names), cols=1, shared_xaxes=True,
        subplot_titles=[_pretty(s) for s in names],
        vertical_spacing=0.07,
    )

    time = df.get("time", df.index)
    for i, sensor in enumerate(names):
        for axis, col in sorted(sensors[sensor], key=lambda t: axis_order.get(t[0], 99)):
            fig.add_trace(
                go.Scattergl(
                    x=time, y=df[col], name=axis.upper(),
                    line=dict(color=AXIS_COLORS.get(axis, "#555"), width=1),
                    legendgroup=axis, showlegend=(i == 0),
                ),
                row=i + 1, col=1,
            )

    unit = "m/s\u00b2" if sensor_type == "accel" else "rad/s"
    fig.update_layout(
        height=max(400, len(names) * 230),
        title_text=f"{'Accelerometer' if sensor_type == 'accel' else 'Gyroscope'} ({unit})",
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time (s)", row=len(names), col=1)
    return fig


def build_label_figure(df, data_type):
    """Build a multi-row figure for joint moments or angles.
    Joints are auto-detected from column name patterns."""
    if data_type == "moment":
        left_cols = [c for c in df.columns if c.endswith("_l_moment")]
        joints = [c.replace("_l_moment", "") for c in left_cols]
        col_fn = lambda joint, side: f"{joint}_{side}_moment"
        title = "Joint Moments (Nm/kg)"
    else:
        joints = []
        for c in df.columns:
            if c.endswith("_l") and c != "time":
                joint = c[:-2]
                if f"{joint}_r" in df.columns:
                    joints.append(joint)
        col_fn = lambda joint, side: f"{joint}_{side}"
        title = "Joint Angles (deg)"

    if not joints:
        return _empty_figure("No data available")

    fig = make_subplots(
        rows=len(joints), cols=1, shared_xaxes=True,
        subplot_titles=[_pretty(j) for j in joints],
        vertical_spacing=0.04,
    )

    time = df.get("time", df.index)
    for i, joint in enumerate(joints):
        for side, style in SIDE_STYLES.items():
            col = col_fn(joint, side)
            if col in df.columns and df[col].notna().any():
                fig.add_trace(
                    go.Scattergl(
                        x=time, y=df[col], name=style["label"],
                        line=dict(color=style["color"], width=1),
                        legendgroup=side, showlegend=(i == 0),
                    ),
                    row=i + 1, col=1,
                )

    fig.update_layout(
        height=max(400, len(joints) * 165),
        title_text=title,
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time (s)", row=len(joints), col=1)
    return fig


def _empty_figure(msg):
    fig = go.Figure()
    fig.update_layout(
        height=200,
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font_size=16, font_color="#999")],
        template="plotly_white",
    )
    return fig


# ── Dash App ──────────────────────────────────────────────────────────────────

S = {
    "page": {
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "maxWidth": "1440px", "margin": "0 auto", "padding": "24px",
        "backgroundColor": "#f7f8fa", "minHeight": "100vh",
    },
    "header": {
        "fontSize": "26px", "fontWeight": "700", "color": "#1a1a2e",
        "marginBottom": "20px", "borderBottom": "3px solid #0066cc", "paddingBottom": "8px",
    },
    "card": {
        "backgroundColor": "white", "borderRadius": "8px", "padding": "14px 20px",
        "marginBottom": "14px", "boxShadow": "0 1px 3px rgba(0,0,0,.08)",
    },
    "inp": {
        "flex": "1", "padding": "8px 12px", "border": "1px solid #d0d0d0",
        "borderRadius": "4px", "fontSize": "14px",
    },
    "btn": {
        "padding": "8px 20px", "border": "none", "borderRadius": "4px",
        "cursor": "pointer", "fontSize": "14px", "fontWeight": "600",
    },
}

BTN_PRIMARY = {**S["btn"], "backgroundColor": "#0066cc", "color": "white"}
BTN_NAV = {**S["btn"], "backgroundColor": "#e8eaed", "color": "#333", "border": "1px solid #ccc"}
BTN_FLAG = {**S["btn"], "backgroundColor": "#d32f2f", "color": "white"}
BTN_UNFLAG = {**S["btn"], "backgroundColor": "#388e3c", "color": "white"}


def create_app():
    app = dash.Dash(__name__, title="Trial Viewer", suppress_callback_exceptions=True)

    app.layout = html.Div(style=S["page"], children=[
        dcc.Store(id="trial-list", data=[]),
        dcc.Store(id="flagged-trials", data=[]),

        html.Div(id="kbd-target", style={"display": "none"}),

        html.H1("Biomechanics Trial Viewer", style=S["header"]),

        # ── Path input ──
        html.Div(style={**S["card"], "display": "flex", "alignItems": "center", "gap": "12px"}, children=[
            html.Label("Dataset Root:", style={"fontWeight": "600", "whiteSpace": "nowrap"}),
            dcc.Input(id="dataset-path", type="text", placeholder="/path/to/dataset",
                      style=S["inp"], debounce=True),
            html.Button("Load", id="load-btn", style=BTN_PRIMARY),
            html.Span(id="load-status", style={"color": "#666", "fontSize": "13px"}),
        ]),

        # ── Navigation bar ──
        html.Div(id="nav-bar", style={**S["card"], "display": "none"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"}, children=[
                html.Button("\u25c0 Prev", id="prev-btn", style=BTN_NAV, n_clicks=0),
                html.Button("Next \u25b6", id="next-btn", style=BTN_NAV, n_clicks=0),
                html.Div(dcc.Dropdown(id="trial-dropdown", clearable=False),
                         style={"flex": "1", "minWidth": "280px"}),
                html.Span(id="trial-counter",
                          style={"fontSize": "14px", "fontWeight": "600", "color": "#555",
                                 "padding": "6px 12px", "backgroundColor": "#f0f0f0",
                                 "borderRadius": "4px", "whiteSpace": "nowrap"}),
                html.Button("Flag as Problematic", id="flag-btn", n_clicks=0, style=BTN_FLAG),
            ]),
            html.Div(
                "Keyboard: \u2190 / \u2192 navigate, F flag/unflag",
                style={"fontSize": "11px", "color": "#aaa", "marginTop": "6px"},
            ),
        ]),

        # ── Status banner (shown when trial is flagged) ──
        html.Div(id="trial-status"),

        # ── Plot area ──
        html.Div(id="plot-area", style={"display": "none"}, children=[
            dcc.Tabs(id="plot-tabs", value="tab-accel", children=[
                dcc.Tab(label="Accelerometer", value="tab-accel"),
                dcc.Tab(label="Gyroscope", value="tab-gyro"),
                dcc.Tab(label="Joint Moments", value="tab-moment"),
                dcc.Tab(label="Joint Angles", value="tab-angle"),
            ]),
            dcc.Loading(type="circle", children=html.Div(id="tab-content")),
        ]),

        # ── Flagged trials panel ──
        html.Div(id="flagged-panel", style={"marginTop": "20px"}),
    ])

    # ── Clientside keyboard handler ──
    app.clientside_callback(
        """
        function(_) {
            if (window._tvKeysReady) return dash_clientside.no_update;
            window._tvKeysReady = true;
            document.addEventListener('keydown', function(e) {
                var t = (e.target.tagName || '').toLowerCase();
                if (t === 'input' || t === 'textarea' || t === 'select') return;
                if (e.key === 'ArrowLeft')  { document.getElementById('prev-btn').click(); e.preventDefault(); }
                if (e.key === 'ArrowRight') { document.getElementById('next-btn').click(); e.preventDefault(); }
                if (e.key === 'f' || e.key === 'F') { document.getElementById('flag-btn').click(); e.preventDefault(); }
            });
            return dash_clientside.no_update;
        }
        """,
        Output("kbd-target", "children"),
        Input("kbd-target", "id"),
    )

    # ── Callbacks ──

    @app.callback(
        Output("trial-list", "data"),
        Output("trial-dropdown", "options"),
        Output("trial-dropdown", "value"),
        Output("load-status", "children"),
        Output("nav-bar", "style"),
        Output("plot-area", "style"),
        Input("load-btn", "n_clicks"),
        State("dataset-path", "value"),
        prevent_initial_call=True,
    )
    def load_dataset(_, path):
        hidden = {**S["card"], "display": "none"}
        if not path or not os.path.isdir(path):
            return [], [], None, "Invalid path.", hidden, {"display": "none"}

        trials = discover_trials(path)
        if not trials:
            return [], [], None, "No trials found.", hidden, {"display": "none"}

        opts = [{"label": t["label"], "value": i} for i, t in enumerate(trials)]
        return (
            trials, opts, 0,
            f"Loaded {len(trials)} trials",
            {**S["card"], "display": "block"},
            {"display": "block"},
        )

    @app.callback(
        Output("trial-dropdown", "value", allow_duplicate=True),
        Input("prev-btn", "n_clicks"),
        Input("next-btn", "n_clicks"),
        State("trial-dropdown", "value"),
        State("trial-list", "data"),
        prevent_initial_call=True,
    )
    def navigate(_, __, current, trials):
        if not trials or current is None:
            return no_update
        n = len(trials)
        if ctx.triggered_id == "prev-btn":
            return max(0, current - 1)
        if ctx.triggered_id == "next-btn":
            return min(n - 1, current + 1)
        return no_update

    @app.callback(
        Output("trial-counter", "children"),
        Output("trial-status", "children"),
        Output("flag-btn", "children"),
        Output("flag-btn", "style"),
        Input("trial-dropdown", "value"),
        Input("flagged-trials", "data"),
        State("trial-list", "data"),
    )
    def update_display(idx, flagged, trials):
        if not trials or idx is None:
            return "", "", "Flag as Problematic", BTN_FLAG

        label = trials[idx]["label"]
        counter = f"{idx + 1} / {len(trials)}"
        flagged = flagged or []
        is_flagged = label in flagged

        if is_flagged:
            status = html.Div(
                f"\u26a0  Flagged as problematic: {label}",
                style={"backgroundColor": "#fff3cd", "border": "1px solid #ffc107",
                       "borderRadius": "4px", "padding": "8px 16px", "color": "#856404",
                       "fontWeight": "600", "fontSize": "14px", "marginBottom": "10px"},
            )
            return counter, status, "\u2605 Unflag Trial", BTN_UNFLAG

        return counter, "", "Flag as Problematic", BTN_FLAG

    @app.callback(
        Output("tab-content", "children"),
        Input("trial-dropdown", "value"),
        Input("plot-tabs", "value"),
        State("trial-list", "data"),
    )
    def update_plots(idx, tab, trials):
        if not trials or idx is None:
            return html.Div("Select a dataset to begin.",
                            style={"padding": "40px", "color": "#999", "textAlign": "center"})

        data = get_trial_data(trials[idx])

        tab_map = {
            "tab-accel":  ("imu",    lambda d: build_imu_figure(d, "accel")),
            "tab-gyro":   ("imu",    lambda d: build_imu_figure(d, "gyro")),
            "tab-moment": ("moment", lambda d: build_label_figure(d, "moment")),
            "tab-angle":  ("angle",  lambda d: build_label_figure(d, "angle")),
        }

        key, builder = tab_map.get(tab, (None, None))
        if key and key in data:
            fig = builder(data[key])
            return dcc.Graph(figure=fig, config={"scrollZoom": True})

        return html.Div("Data file not found for this trial.",
                         style={"padding": "40px", "color": "#999", "textAlign": "center"})

    @app.callback(
        Output("flagged-trials", "data"),
        Input("flag-btn", "n_clicks"),
        State("trial-dropdown", "value"),
        State("trial-list", "data"),
        State("flagged-trials", "data"),
        prevent_initial_call=True,
    )
    def toggle_flag(_, idx, trials, flagged):
        if not trials or idx is None:
            return no_update
        label = trials[idx]["label"]
        flagged = list(flagged or [])
        if label in flagged:
            flagged.remove(label)
        else:
            flagged.append(label)
        return flagged

    @app.callback(
        Output("flagged-panel", "children"),
        Input("flagged-trials", "data"),
    )
    def update_flagged_panel(flagged):
        flagged = flagged or []

        panel_style = {
            **S["card"],
            "border": f"2px solid {'#d32f2f' if flagged else '#e0e0e0'}",
        }

        if not flagged:
            return html.Div(style=panel_style, children=[
                html.H3("Flagged Trials",
                         style={"margin": "0 0 6px", "fontSize": "16px", "color": "#888"}),
                html.P("No trials flagged yet.",
                       style={"margin": "0", "color": "#bbb", "fontSize": "14px"}),
            ])

        items = [html.Li(t, style={"padding": "2px 0", "fontSize": "14px"}) for t in sorted(flagged)]
        copyable = "\n".join(sorted(flagged))

        return html.Div(style=panel_style, children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "alignItems": "center"}, children=[
                html.H3(f"Flagged Trials ({len(flagged)})",
                         style={"margin": "0", "fontSize": "16px", "color": "#d32f2f"}),
            ]),
            html.Ul(items, style={"margin": "10px 0 0", "paddingLeft": "22px"}),
            html.Hr(style={"margin": "12px 0", "border": "none", "borderTop": "1px solid #eee"}),
            html.Details([
                html.Summary("Copy-friendly list",
                             style={"cursor": "pointer", "fontSize": "13px", "color": "#888"}),
                dcc.Textarea(
                    value=copyable, readOnly=True,
                    style={"width": "100%", "height": "120px", "marginTop": "6px",
                           "fontFamily": "monospace", "fontSize": "13px",
                           "border": "1px solid #ddd", "borderRadius": "4px", "padding": "8px"},
                ),
            ]),
        ])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    port = 8050
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    app = create_app()
    print(f"\n  Trial Viewer: http://localhost:{port}\n")
    app.run(debug=False, port=port, host="0.0.0.0")


if __name__ == "__main__":
    main()
