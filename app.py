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
    """
    Reduce thermocouple noise using
    rolling median + rolling mean
    """
    return (
        series
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )


def clean_and_smooth(series, apply_smoothing, window):
    """
    Full TC conditioning pipeline:
    1. Numeric conversion
    2. Remove negative values
    3. Remove statistical outliers (IQR)
    4. Interpolate gaps
    5. Smooth / linearize signal
    """
    series = pd.to_numeric(series, errors="coerce")

    # Hard physical rule: temperature cannot be negative
    series = series.where(series >= 0)

    # IQR-based outlier removal
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    series = series.where((series >= lower) & (series <= upper))

    # Interpolate missing values
    series = series.interpolate(method="linear", limit_direction="both")

    # Noise smoothing / linearization
    if apply_smoothing:
        series = smooth_signal(series, window)

    return series

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.file_uploader("Choose an INSTR CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV (skip first 26 rows)
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

    # Remove alarm columns
    df = df.loc[:, ~df.columns.str.contains("Alarm")]

    # Fix timestamp format
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
        value=11,
        help="Larger window = smoother but slower response"
    )

    if selected_sensors:
        plot_df = df[["Time"] + selected_sensors].copy()

        # Apply conditioning per sensor
        for sensor in selected_sensors:
            plot_df[sensor] = pd.to_numeric(plot_df[sensor], errors="coerce")

            if remove_outliers:
                plot_df[sensor] = clean_and_smooth(
                    plot_df[sensor],
                    apply_smoothing=smooth_noise,
                    window=window_size
                )

        # Melt for limit initialization
        plot_df_melted = plot_df.melt(
            id_vars="Time",
            var_name="Sensor",
            value_name="Temperature"
        )

        min_temp = float(plot_df_melted["Temperature"].min())
        max_temp = float(plot_df_melted["Temperature"].max())

        lower_limit = st.number_input(
            "Initial Lower Limit",
            value=min_temp
        )
        upper_limit = st.number_input(
            "Initial Upper Limit",
            value=max_temp
        )

        # ---------------------------
        # Plotly figure
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

        # Lower limit line
        fig.add_shape(
            type="line",
            x0=plot_df["Time"].min(),
            x1=plot_df["Time"].max(),
            y0=lower_limit,
            y1=lower_limit,
            line=dict(color="red", width=2, dash="dash"),
            editable=True
        )

        # Upper limit line
        fig.add_shape(
            type="line",
            x0=plot_df["Time"].min(),
            x1=plot_df["Time"].max(),
            y0=upper_limit,
            y1=upper_limit,
            line=dict(color="red", width=2, dash="dash"),
            editable=True
        )

        fig.update_layout(
            title="Bake Temperature Profile",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            dragmode="drawline"
        )

        st.plotly_chart(fig, use_container_width=True)
