# Trial Viewer

Interactive web app for browsing biomechanics trials: IMU signals (accelerometer, gyroscope) and joint inverse kinematics / inverse dynamics (angles, moments). Built with [Dash](https://dash.plotly.com/) and Plotly.

## Requirements

- Python 3.10+

## Install

From this directory:

```bash
pip install -e .
```

Dependencies: `dash`, `pandas`, `plotly`, `h5py`, `scipy`.

## Run

```bash
trial-viewer
```

Optional port (default `8050`):

```bash
trial-viewer 8765
```

Or:

```bash
python -m trial_viewer
python -m trial_viewer 8765
```

Open the URL printed in the terminal (e.g. `http://localhost:8050`). The server binds to `0.0.0.0`.

## Usage

1. Enter the **dataset root** path and click **Load**.
2. Use **Prev** / **Next**, the trial dropdown, or keyboard shortcuts to move between trials.
3. Tabs: **Accelerometer**, **Gyroscope**, **Joint Moments**, **Joint Angles**.
4. **Flag as Problematic** marks the current trial; flagged labels appear in the panel at the bottom (with a copy-friendly list).

**Keyboard** (when focus is not in a text field):

| Key | Action |
|-----|--------|
| ← / → | Previous / next trial |
| F | Toggle flag on current trial |

Displayed time series are smoothed with a **4 Hz zero-phase lowpass** for viewing only; raw data in files is unchanged.

## Supported dataset layouts

1. **Paired filesystem + HDF5** — Subject folders under a named root (e.g. `Camargo`, `MetaMobility`, `Scherpereel`, `Molinaro_Scherpereel`) with `Snnn/<condition>/trial_k/` and `condition_meta.json`; per-subject `Snnn.h5` in a sibling `*_h5` folder (e.g. `Camargo_h5`).
2. **HDF5-only** — Point the root at `*_h5` or a flat folder of `Snnn.h5` files (e.g. MeMo-style exports); trials are discovered from HDF5 groups.
3. **CSV layout (legacy)** — Directories containing `Input/imu_data.csv` and optionally `Label/joint_moment.csv`, `Label/joint_angle.csv`.

If loading fails, confirm the path exists and matches one of these layouts.

## License

See `LICENSE` in this repository.
