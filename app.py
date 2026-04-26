import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.detection_system import ElectricityTheftDetector
import time

st.set_page_config(page_title="Electricity Theft Detection", layout="wide")

@st.cache_resource
def load_model():
    return ElectricityTheftDetector()

detector = load_model()

st.title("⚡ Electricity Theft Detection System")
st.markdown("---")

# Sidebar for single reading detection
st.sidebar.header("Single Reading Analysis")
consumption = st.sidebar.slider("Consumption (kWh)", 0.0, 10.0, 2.5)
voltage = st.sidebar.slider("Voltage (V)", 200.0, 260.0, 230.0)
current = st.sidebar.slider("Current (A)", 0.0, 50.0, 10.0)
power_factor = st.sidebar.slider("Power Factor", 0.4, 1.0, 0.9)
hour = st.sidebar.slider("Hour (0-23)", 0, 23, 18)

if st.sidebar.button("🔍 Detect Theft"):
    with st.spinner("Analyzing..."):
        result = detector.detect_theft(
            consumption_kwh=consumption,
            voltage=voltage,
            current=current,
            power_factor=power_factor,
            hour=hour
        )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Theft Detected", "YES" if result['is_theft'] else "NO",
                 delta=f"{result['theft_probability']*100:.1f}%")
    with col2:
        st.metric("Confidence", f"{result['theft_probability']*100:.1f}%")
    with col3:
        st.metric("RF Prediction", "THEFT" if result['rf_prediction'] else "NORMAL")
    with col4:
        st.metric("Consumption", f"{result['consumption_kwh']:.2f} kWh")

# Main dashboard
tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📈 Historical Analysis", "⚙️ Upload Data"])

with tab1:
    st.header("Live Monitoring Dashboard")
    
    # Simulated real-time data
    placeholder = st.empty()
    
    for _ in range(100):
        time.sleep(0.1)
        sim_consumption = 2.5 + np.random.normal(0, 0.5)
        sim_voltage = 230 + np.random.normal(0, 3)
        sim_current = sim_consumption * 4.3
        sim_pf = 0.9 + np.random.normal(0, 0.05)
        
        result = detector.detect_theft(sim_consumption, sim_voltage, sim_current, sim_pf)
        
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Consumption", f"{sim_consumption:.2f} kWh")
            with col2:
                st.metric("Status", "🚨 THEFT DETECTED" if result['is_theft'] else "✅ Normal")
            with col3:
                progress = st.progress(result['theft_probability'])
                st.metric("Theft Risk", f"{result['theft_probability']*100:.1f}%")

with tab2:
    st.header("Historical Analysis")
    
    # Load sample data
    @st.cache_data
    def load_sample_data():
        df = pd.read_csv('data/full_dataset.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return detector.batch_detect(df)
    
    df_analysis = load_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        theft_rate = df_analysis['is_theft'].mean() * 100
        st.metric("Theft Detection Rate", f"{theft_rate:.1f}%")
    
    with col2:
        high_risk = (df_analysis['theft_probability'] > 0.8).sum()
        st.metric("High Risk Readings", high_risk)
    
    fig = px.scatter(df_analysis, x='timestamp', y='consumption_kwh', 
                    color='is_theft', title="Consumption Patterns",
                    color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose CSV file", type='csv')
    
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df_uploaded.head())
        
        if st.button("🚀 Analyze Uploaded Data"):
            with st.spinner("Analyzing..."):
                results = detector.batch_detect(df_uploaded)
                
                st.success("Analysis complete!")
                st.dataframe(results[['consumption_kwh', 'voltage', 'theft_probability', 'is_theft']])
                
                fig = px.histogram(results, x='theft_probability', 
                                 color='is_theft', nbins=50,
                                 title="Theft Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit, Scikit-learn & Pandas**")