import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
MASTER_CSV = 'master_dataset.csv'
V_REF = 5.0        
ADC_MAX = 1023     

# --- Page Config (Must be first) ---
st.set_page_config(
    page_title="SensorEdge Pro | Enterprise Dashboard", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }
    /* Success Messages */
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logic & Helper Functions ---
def load_master():
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            # Normalize column names
            df.rename(columns=lambda x: x.strip(), inplace=True)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except:
            return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])

def standardize_columns(df):
    """Smartly renames columns to standard format."""
    df.columns = df.columns.str.strip()
    col_map = {
        'time': 'Timestamp', 'date': 'Timestamp', 'datetime': 'Timestamp', 'Time': 'Timestamp',
        'sensor': 'Sensor_Name', 'sensor_name': 'Sensor_Name', 'Sensor': 'Sensor_Name',
        'voltage': 'Voltage_V', 'Voltage': 'Voltage_V', 'volts': 'Voltage_V',
        'adc': 'ADC_Value', 'ADC': 'ADC_Value'
    }
    df.rename(columns=col_map, inplace=True)
    return df

def calculate_analytics(df):
    """Returns a dictionary of professional metrics."""
    metrics = {
        "hw_acc": 0.0, "ai_acc": 0.0, "status": "No Data", "color": "off"
    }
    
    if df.empty or 'Voltage_V' not in df.columns:
        return metrics

    # 1. Hardware Integrity (ADC Check)
    try:
        df_clean = df.dropna(subset=['Voltage_V', 'ADC_Value'])
        if not df_clean.empty:
            expected_adc = (df_clean['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            matches = (abs(df_clean['ADC_Value'] - expected_adc) <= 1)
            metrics["hw_acc"] = matches.mean() * 100
    except:
        pass

    # 2. AI Classification Score
    try:
        if df_clean['Sensor_Name'].nunique() >= 2:
            X = df_clean[['Voltage_V']]
            y = df_clean['Sensor_Name']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=20, random_state=42)
            clf.fit(X_train, y_train)
            metrics["ai_acc"] = accuracy_score(y_test, clf.predict(X_test)) * 100
            metrics["status"] = "System Healthy"
            metrics["color"] = "normal"
        else:
            metrics["status"] = "Calibrating (Need >1 Sensor)"
            metrics["color"] = "off"
    except:
        metrics["status"] = "Model Error"
        metrics["color"] = "off"
        
    return metrics

def to_excel(df):
    output = io.BytesIO()
    
    def highlight_rows(row):
        status = row.get('Status', '')
        if status == 'New': return ['background-color: #d4edda'] * len(row) # Soft Green
        elif status == 'Overlap': return ['background-color: #f8d7da'] * len(row) # Soft Red
        return [''] * len(row)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.style.apply(highlight_rows, axis=1).to_excel(writer, index=False, sheet_name='SensorData')
    return output.getvalue()

# --- Load Data ---
master_df = load_master()
analytics = calculate_analytics(master_df)

# --- SIDEBAR: Controls & Inputs ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2906/2906274.png", width=50) # Generic Sensor Icon
    st.title("Control Panel")
    st.markdown("---")
    
    # Upload Section
    st.subheader("📤 Ingest Data")
    uploaded_file = st.file_uploader("Drop CSV Here", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            with st.spinner("Processing & Merging..."):
                # Load & Standardize
                new_df = pd.read_csv(uploaded_file)
                new_df = standardize_columns(new_df)
                
                # Check Requirements
                if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
                    st.error("❌ Invalid File Format")
                else:
                    # Logic
                    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'], errors='coerce')
                    new_df = new_df.dropna(subset=['Timestamp'])
                    new_df['Sensor_Name'] = new_df['Sensor_Name'].astype(str).str.strip()
                    if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
                    
                    new_df['Status'] = 'New'
                    
                    if not master_df.empty:
                        master_df['Status'] = 'Historical'
                        # Overlap Detection
                        master_keys = set(zip(master_df['Timestamp'], master_df['Sensor_Name']))
                        new_keys = set(zip(new_df['Timestamp'], new_df['Sensor_Name']))
                        overlap = master_keys.intersection(new_keys)
                        
                        master_df['Status'] = master_df.apply(
                            lambda x: 'Overlap' if (x['Timestamp'], x['Sensor_Name']) in overlap else 'Historical', axis=1
                        )

                    # Merge & Save
                    combined = pd.concat([master_df, new_df]).sort_values('Timestamp')
                    final = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
                    final.to_csv(MASTER_CSV, index=False)
                    
                    st.success("Sync Complete!")
                    time.sleep(1)
                    st.rerun()
        except Exception as e:
            st.error(f"Sync Failed: {e}")

    st.markdown("---")
    st.subheader("⚙️ System")
    if st.button("Factory Reset Database", type="secondary"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.rerun()

    st.info(f"Last Update:\n{datetime.now().strftime('%H:%M:%S')}")

# --- MAIN DASHBOARD ---

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("SensorEdge Analytics")
    st.markdown("Real-time monitoring and anomaly detection system.")
with col_h2:
    # Status Badge
    if analytics["status"] == "System Healthy":
        st.success(f"● {analytics['status']}")
    else:
        st.warning(f"● {analytics['status']}")

# KPI Cards (Top Row)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total Records", f"{len(master_df):,}", delta="Live Count")

if not master_df.empty and 'Timestamp' in master_df.columns:
    duration = master_df['Timestamp'].max() - master_df['Timestamp'].min()
    hours = duration.total_seconds() / 3600
    kpi2.metric("Data Span", f"{hours:.1f} hrs", "Time Coverage")
else:
    kpi2.metric("Data Span", "0 hrs")

kpi3.metric("Hardware Fidelity", f"{analytics['hw_acc']:.1f}%", help="ADC vs Voltage consistency")
kpi4.metric("AI Confidence", f"{analytics['ai_acc']:.1f}%", help="Sensor Classification Accuracy")

st.markdown("---")

# Content Tabs
tab_charts, tab_data = st.tabs(["📊 Analytics Visuals", "📋 Data Inspector"])

with tab_charts:
    if not master_df.empty and 'Voltage_V' in master_df.columns:
        col_c1, col_c2 = st.columns([3, 1])
        
        with col_c2:
            st.subheader("Filter View")
            sensors = master_df['Sensor_Name'].unique().tolist()
            sel_sensors = st.multiselect("Select Sensors", sensors, default=sensors)
        
        with col_c1:
            st.subheader("Voltage Telemetry")
            filtered = master_df[master_df['Sensor_Name'].isin(sel_sensors)]
            chart_data = filtered.pivot_table(index='Timestamp', columns='Sensor_Name', values='Voltage_V', aggfunc='first')
            st.line_chart(chart_data, height=350)
    else:
        st.info("Awaiting Data Upload...")

with tab_data:
    st.subheader("Master Ledger")
    
    if not master_df.empty:
        # Professional Toolbar
        col_t1, col_t2 = st.columns([4, 1])
        with col_t1:
            filter_status = st.radio("Record Status:", ["All", "New (Green)", "Overlap (Red)", "Historical"], horizontal=True)
        with col_t2:
             st.download_button(
                "📥 Export Report", 
                data=to_excel(master_df), 
                file_name="sensor_report.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Filter Logic
        view_df = master_df.copy()
        if filter_status == "New (Green)": view_df = view_df[view_df['Status'] == 'New']
        elif filter_status == "Overlap (Red)": view_df = view_df[view_df['Status'] == 'Overlap']
        elif filter_status == "Historical": view_df = view_df[view_df['Status'] == 'Historical']

        # Dataframe Configuration (The "Pro" Look)
        st.dataframe(
            view_df,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm:ss"),
                "Voltage_V": st.column_config.ProgressColumn("Voltage", format="%.2f V", min_value=0, max_value=5),
                "ADC_Value": st.column_config.NumberColumn("ADC", format="%d"),
                "Status": st.column_config.TextColumn("Sync Status"),
            },
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.write("No records found in database.")
