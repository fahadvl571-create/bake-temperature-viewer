import os
import time
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Bake Temperature Viewer", layout="wide")
st.title("Bake Temperature Viewer")

# ---------------------------
# Signal processing helpers
# ---------------------------
def smooth_signal(series, window):
    return (
        series
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )


def clean_and_smooth(series, apply_smoothing, window):
    series = pd.to_numeric(series, errors="coerce")

    # Hard physical rule: negative is invalid
    series = series.where(series >= 0)

    # IQR outlier removal
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    series = series.where((series >= lower) & (series <= upper))

    # Fill gaps
    series = series.interpolate(method="linear", limit_direction="both")

    # Smooth noise
    if apply_smoothing:
        series = smooth_signal(series, window)

    return series


# ---------------------------
# CSV loader
# ---------------------------
def read_instr_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(
        pd.io.common.BytesIO(file_bytes),
        header=None,
        skiprows=26,
        encoding="utf-16",
        sep="\t",
    )

    df = df[0].str.split(",", expand=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains("Alarm")]

    df["Time"] = df["Time"].str.replace(r":(?=\d{3}$)", ".", regex=True)
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%Y %H:%M:%S.%f")

    return df


def read_instr_csv_from_path(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        file_bytes = f.read()
    return read_instr_csv_from_bytes(file_bytes)


# ---------------------------
# Projection helper
# ---------------------------
def estimate_crossing_time(time_series: pd.Series, temp_series: pd.Series, target_c: float, lookback_samples: int = 30):
    """
    Estimate when temp crosses target using a linear fit over the most recent samples.
    Returns: (t_cross: pd.Timestamp or None, slope_c_per_sec: float or None)
    """
    ts = pd.to_datetime(time_series)
    ys = pd.to_numeric(temp_series, errors="coerce")

    mask = ts.notna() & ys.notna()
    ts = ts[mask]
    ys = ys[mask]

    if len(ts) < 5:
        return None, None

    # If already above target at last point, crossing is now
    if ys.iloc[-1] >= target_c:
        return ts.iloc[-1], 0.0

    # Take recent window
    ts_win = ts.iloc[-lookback_samples:]
    ys_win = ys.iloc[-lookback_samples:]

    if len(ts_win) < 5:
        return None, None

    # Convert time to seconds relative to start of window
    t0 = ts_win.iloc[0]
    x = (ts_win - t0).dt.total_seconds().to_numpy(dtype=float)
    y = ys_win.to_numpy(dtype=float)

    # If time isn't increasing, bail
    if np.allclose(x.max(), 0):
        return None, None

    # Linear fit: y = m*x + b
    m, b = np.polyfit(x, y, 1)

    # Need a positive slope to project forward
    if m <= 1e-9:
        return None, m

    # Solve for x when y = target
    x_cross = (target_c - b) / m
    if x_cross < 0:
        # Would imply it crossed before the window, but last point is below target;
        # treat as not reliably projectable.
        return None, m

    t_cross = t0 + pd.to_timedelta(float(x_cross), unit="s")
    return t_cross, m


# ---------------------------
# Sidebar: refresh + source
# ---------------------------
with st.sidebar:
    st.header("Live refresh")
    enable_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_seconds = st.number_input(
        "Refresh interval (sec)",
        min_value=5,
        max_value=300,
        value=180,
        step=5,
    )

    # OPTION 1: Built-in refresh (no extra packages)
    if enable_refresh:
        st.markdown(
            f"<meta http-equiv='refresh' content='{int(refresh_seconds)}'>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.header("Data source")
    source_mode = st.radio(
        "Choose source",
        ["Upload once then auto-refresh", "Read from fixed path (server-side)"],
        index=0,
    )

st.caption("Tip: On Streamlit Cloud, 'fixed path' means a path on the cloud server, not your local PC.")

# ---------------------------
# Choose data source
# ---------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DEFAULT_SAVED_PATH = os.path.join(DATA_DIR, "latest_instr.csv")

df = None
status_line = None

if source_mode == "Read from fixed path (server-side)":
    fixed_path = st.text_input("CSV path (server-side)", value=DEFAULT_SAVED_PATH)

    if fixed_path and os.path.exists(fixed_path):
        df = read_instr_csv_from_path(fixed_path)
        status_line = f"Reading: {fixed_path} | Last modified: {time.ctime(os.path.getmtime(fixed_path))}"
    else:
        st.warning("File not found at that path (on the server where Streamlit is running).")

else:
    uploaded_file = st.file_uploader("Choose an INSTR CSV file (upload once)", type="csv")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        # Save it so reruns can reuse it without re-upload
        with open(DEFAULT_SAVED_PATH, "wb") as f:
            f.write(file_bytes)

        df = read_instr_csv_from_bytes(file_bytes)
        status_line = f"Uploaded & saved as: {DEFAULT_SAVED_PATH}"

    elif os.path.exists(DEFAULT_SAVED_PATH):
        df = read_instr_csv_from_path(DEFAULT_SAVED_PATH)
        status_line = f"Using saved file: {DEFAULT_SAVED_PATH} | Last modified: {time.ctime(os.path.getmtime(DEFAULT_SAVED_PATH))}"
    else:
        st.info("Upload the file once. After that, auto-refresh will reuse the saved copy.")

if status_line:
    st.write(status_line)

if df is None:
    st.stop()

# ---------------------------
# UI controls
# ---------------------------
temp_columns = [col for col in df.columns if col not in ["Scan", "Time"]]

selected_sensors = st.multiselect(
    "Select thermocouples to plot",
    temp_columns,
    default=temp_columns,
)

remove_outliers = st.checkbox("Remove outliers & negative values", value=True)
smooth_noise = st.checkbox("Smooth / linearize signal (reduce TC noise)", value=True)

window_size = st.slider(
    "Smoothing window (samples)",
    min_value=3,
    max_value=51,
    step=2,
    value=11,
)

# Soak temperature window
st.subheader("Soak condition")
soak_low = st.number_input("Lower soak limit (°C)", value=110.0)
soak_high = st.number_input("Upper soak limit (°C)", value=150.0)

# Projection tuning
st.subheader("Ramp-up / projection settings")
lookback_samples = st.slider("Projection lookback (samples)", min_value=10, max_value=120, value=30, step=5)

if not selected_sensors:
    st.stop()

plot_df = df[["Time"] + selected_sensors].copy()

for sensor in selected_sensors:
    plot_df[sensor] = pd.to_numeric(plot_df[sensor], errors="coerce")
    if remove_outliers:
        plot_df[sensor] = clean_and_smooth(
            plot_df[sensor],
            apply_smoothing=smooth_noise,
            window=window_size,
        )

# ---------------------------
# Ramp-up time: time for ALL TCs to reach 110C (soak_low)
# ---------------------------
temp_only = plot_df[selected_sensors]

# First time ALL sensors are >= soak_low
all_reached_mask = temp_only.apply(lambda row: row.ge(soak_low).all(), axis=1)

t_start = plot_df["Time"].iloc[0]
t_last = plot_df["Time"].iloc[-1]

t_all_reached = None
if all_reached_mask.any():
    first_idx = all_reached_mask.idxmax()  # first True index
    t_all_reached = plot_df.loc[first_idx, "Time"]

ramp_up_seconds = None
if t_all_reached is not None and pd.notna(t_all_reached) and pd.notna(t_start):
    ramp_up_seconds = (t_all_reached - t_start).total_seconds()

# ---------------------------
# Projection time left for ALL TCs to reach 110C (if not reached yet)
# ---------------------------
projected_t_all = None
projection_note = None

if t_all_reached is None:
    per_sensor_cross = []
    bad_sensors = []

    for sensor in selected_sensors:
        t_cross, slope = estimate_crossing_time(
            plot_df["Time"],
            plot_df[sensor],
            target_c=soak_low,
            lookback_samples=int(lookback_samples),
        )

        if t_cross is None:
            bad_sensors.append(sensor)
        else:
            per_sensor_cross.append(t_cross)

    if per_sensor_cross:
        projected_t_all = max(per_sensor_cross)  # slowest sensor dominates
    else:
        projected_t_all = None

    if bad_sensors:
        projection_note = f"Projection unavailable for: {', '.join(bad_sensors)} (not enough data or not heating up)."

# ---------------------------
# Calculate soak time (ALL sensors within soak window)
# ---------------------------
in_soak = temp_only.apply(lambda row: row.between(soak_low, soak_high).all(), axis=1)
time_delta = plot_df["Time"].diff().dt.total_seconds().fillna(0)

soak_seconds = (time_delta * in_soak).sum()
soak_hours = soak_seconds / 3600

# ---------------------------
# Display metrics
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if ramp_up_seconds is None:
        st.metric(
            label=f"Ramp-Up time (ALL TCs to ≥ {soak_low:.0f}°C)",
            value="Not reached yet",
        )
    else:
        st.metric(
            label=f"Ramp-Up time (ALL TCs to ≥ {soak_low:.0f}°C)",
            value=f"{ramp_up_seconds/3600:.2f} hours",
        )

with col2:
    if t_all_reached is not None:
        st.metric(
            label=f"Projection left to ALL reach {soak_low:.0f}°C",
            value="0.00 hours",
        )
    else:
        if projected_t_all is None or pd.isna(projected_t_all):
            st.metric(
                label=f"Projection left to ALL reach {soak_low:.0f}°C",
                value="N/A",
            )
        else:
            seconds_left = (projected_t_all - t_last).total_seconds()
            if seconds_left < 0:
                seconds_left = 0
            st.metric(
                label=f"Projection left to ALL reach {soak_low:.0f}°C",
                value=f"{seconds_left/3600:.2f} hours",
            )

with col3:
    st.metric(
        label=f"Total time ALL TCs between {soak_low:.0f}–{soak_high:.0f} °C",
        value=f"{soak_hours:.2f} hours",
    )

if projection_note:
    st.info(projection_note)

# ---------------------------
# Plot
# ---------------------------
fig = go.Figure()

for sensor in selected_sensors:
    fig.add_trace(
        go.Scatter(
            x=plot_df["Time"],
            y=plot_df[sensor],
            mode="lines",
            name=sensor,
        )
    )

# Soak lines
fig.add_shape(
    type="line",
    x0=plot_df["Time"].min(),
    x1=plot_df["Time"].max(),
    y0=soak_low,
    y1=soak_low,
    line=dict(color="green", dash="dot"),
)

fig.add_shape(
    type="line",
    x0=plot_df["Time"].min(),
    x1=plot_df["Time"].max(),
    y0=soak_high,
    y1=soak_high,
    line=dict(color="green", dash="dot"),
)

# Optional: add a vertical line when ALL reached soak_low
if t_all_reached is not None:
    fig.add_shape(
        type="line",
        x0=t_all_reached,
        x1=t_all_reached,
        y0=float(np.nanmin(temp_only.to_numpy(dtype=float))),
        y1=float(np.nanmax(temp_only.to_numpy(dtype=float))),
        line=dict(color="black", dash="dash"),
    )

fig.update_layout(
    title="Bake Temperature Profile",
    xaxis_title="Time",
    yaxis_title="Temperature (°C)",
)

st.plotly_chart(fig, use_container_width=True)
