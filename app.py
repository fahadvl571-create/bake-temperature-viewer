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

    # Hard physical rule
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
# Upload CSV
# ---------------------------
uploaded_file = st.file_uploader("Choose an INSTR CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file,
        header=None,
        skiprows=26,
        encoding="utf-16",
        sep="\t"
    )

    df = df[0].str.split(",", expand=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains("Alarm")]

    df["Time"] = df["Time"].str.replace(r":(?=\d{3}$)", ".", regex=True)
    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%m/%d/%Y %H:%M:%S.%f"
    )

    # ---------------------------
    # UI controls
    # ---------------------------
    temp_columns = [col for col in df.columns if col not in ["Scan", "Time"]]

    selected_sensors = st.multiselect(
        "Select thermocouples to plot",
        temp_columns,
        default=temp_columns
    )

    remove_outliers = st.checkbox(
        "Remove outliers & negative values",
        value=True
    )

    smooth_noise = st.checkbox(
        "Smooth / linearize signal (reduce TC noise)",
        value=True
    )

    window_size = st.slider(
        "Smoothing window (samples)",
        min_value=3,
        max_value=51,
        step=2,
        value=11
    )

    # Soak temperature window
    st.subheader("Soak condition")
    soak_low = st.number_input("Lower soak limit (°C)", value=110.0)
    soak_high = st.number_input("Upper soak limit (°C)", value=150.0)

    if selected_sensors:
        plot_df = df[["Time"] + selected_sensors].copy()

        for sensor in selected_sensors:
            plot_df[sensor] = pd.to_numeric(plot_df[sensor], errors="coerce")
            if remove_outliers:
                plot_df[sensor] = clean_and_smooth(
                    plot_df[sensor],
                    apply_smoothing=smooth_noise,
                    window=window_size
                )

        # ---------------------------
        # Calculate soak time
        # ---------------------------
        temp_only = plot_df[selected_sensors]

        # Condition: ALL sensors within range
        in_soak = temp_only.apply(
            lambda row: row.between(soak_low, soak_high).all(),
            axis=1
        )

        # Time delta between samples (seconds)
        time_delta = plot_df["Time"].diff().dt.total_seconds().fillna(0)

        soak_seconds = (time_delta * in_soak).sum()
        soak_hours = soak_seconds / 3600

        # Display result
        st.metric(
            label=f"Total time ALL TCs between {soak_low}–{soak_high} °C",
            value=f"{soak_hours:.2f} hours"
        )

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
                    name=sensor
                )
            )

        fig.add_shape(
            type="line",
            x0=plot_df["Time"].min(),
            x1=plot_df["Time"].max(),
            y0=soak_low,
            y1=soak_low,
            line=dict(color="green", dash="dot")
        )

        fig.add_shape(
            type="line",
            x0=plot_df["Time"].min(),
            x1=plot_df["Time"].max(),
            y0=soak_high,
            y1=soak_high,
            line=dict(color="green", dash="dot")
        )

        fig.update_layout(
            title="Bake Temperature Profile",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)"
        )

        st.plotly_chart(fig, use_container_width=True)
