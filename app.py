import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Bake Temperature Viewer", layout="wide")
st.title("Bake Temperature Viewer")

# ---------------------------
# Upload CSV
# ---------------------------
uploaded_file = st.file_uploader("Choose an INSTR CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV (skip first 26 rows)
    df = pd.read_csv(uploaded_file, header=None, skiprows=26, encoding="utf-16", sep="\t")
    df = df[0].str.split(",", expand=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains("Alarm")]
    df['Time'] = df['Time'].str.replace(r":(?=\d{3}$)", ".", regex=True)
    df['Time'] = pd.to_datetime(df['Time'], format="%m/%d/%Y %H:%M:%S.%f")
    
    # Sensor selection
    temp_columns = [col for col in df.columns if col not in ['Scan', 'Time']]
    selected_sensors = st.multiselect("Select thermocouples to plot", temp_columns, default=temp_columns)
    
    if selected_sensors:
        plot_df = df[['Time'] + selected_sensors].copy()
        plot_df_melted = plot_df.melt(id_vars='Time', var_name='Sensor', value_name='Temperature')
        plot_df_melted['Temperature'] = pd.to_numeric(plot_df_melted['Temperature'], errors='coerce')

        # Initial limit lines
        min_temp = float(plot_df_melted['Temperature'].min())
        max_temp = float(plot_df_melted['Temperature'].max())
        lower_limit = st.number_input("Initial Lower Limit", value=min_temp)
        upper_limit = st.number_input("Initial Upper Limit", value=max_temp)

        # ---------------------------
        # Plotly figure
        # ---------------------------
        fig = go.Figure()

        # Add lines for each sensor
        for sensor in selected_sensors:
            fig.add_trace(go.Scatter(
                x=plot_df['Time'],
                y=pd.to_numeric(plot_df[sensor], errors='coerce'),
                mode='lines',
                name=sensor
            ))

        # Add draggable limit lines
        fig.add_shape(
            type="line",
            x0=plot_df['Time'].min(),
            x1=plot_df['Time'].max(),
            y0=lower_limit,
            y1=lower_limit,
            line=dict(color="red", width=2, dash="dash"),
            editable=True,  # makes it draggable
            name="Lower Limit"
        )

        fig.add_shape(
            type="line",
            x0=plot_df['Time'].min(),
            x1=plot_df['Time'].max(),
            y0=upper_limit,
            y1=upper_limit,
            line=dict(color="red", width=2, dash="dash"),
            editable=True,  # makes it draggable
            name="Upper Limit"
        )

        fig.update_layout(
            title="Bake Temperature Profile",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            dragmode="drawline",  # allow line dragging
        )


        st.plotly_chart(fig, use_container_width=True)
