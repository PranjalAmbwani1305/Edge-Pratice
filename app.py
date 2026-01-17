import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Configuration ---
MASTER_CSV = 'master_dataset.csv'
V_REF = 5.0        
ADC_MAX = 1023     

# --- 1. Page Config & Custom CSS ---
st.set_page_config(
    page_title="SensorEdge Enterprise", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Card" look and cleaner fonts
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Core Logic Functions ---
@st.cache_data(ttl=60)
def load_master():
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except:
            pass
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])

def generate_spec_data():
    """Generates the Golden Dataset for EXACTLY 3 Sensors."""
    start = datetime(2024, 1, 1, 0, 0, 0)
    duration = 24 * 3600
    
    def make(name, interval, v_min, v_max):
        times = [start + timedelta(seconds=i) for i in range(0, duration, interval)]
        volts = np.random.uniform(v_min, v_max, len(times))
        adcs = (volts / V_REF * ADC_MAX).astype(int)
        return pd.DataFrame({
            'Timestamp': times, 'Sensor_Name': name, 
            'Voltage_V': np.round(volts, 4), 'ADC_Value': adcs, 'Status': 'Historical'
        })
    
    # 1. Temperature (Every 2s)
    df1 = make("Temperature", 2, 2.0, 4.0)
    
    # 2. Moisture (Every 4h)
    df2 = make("Moisture", 14400, 1.2, 3.0)
    
    # 3. Light (Every 5s)
    df3 = make("Light", 5, 0.0, 5.0)
    
    return pd.concat([df1, df2, df3]).sort_values('Timestamp').reset_index(drop=True)

def process_merge_fast(master_df, new_df):
    """High-Performance Merge Logic."""
    new_df.columns = new_df.columns.str.strip().str.title()
    rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 
                  'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 'Adc': 'ADC_Value'}
    new_df.rename(columns=rename_map, inplace=True)
    
    if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
        return None, "❌ Invalid Format: Missing Timestamp or Sensor_Name"

    new_df['Status'] = 'New'
    
    if not master_df.empty:
        master_df['Status'] = 'Historical'
        m_idx = pd.MultiIndex.from_frame(master_df[['Timestamp', 'Sensor_Name']])
        n_idx = pd.MultiIndex.from_frame(new_df[['Timestamp', 'Sensor_Name']])
        master_df.loc[m_idx.isin(n_idx), 'Status'] = 'Overlap'

    combined = pd.concat([master_df, new_df])
    combined['Timestamp'] = pd.to_datetime(combined['Timestamp'])
    final = combined.sort_values('Timestamp').drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    return final, "Success"

def get_analytics(df):
    """Calculates granular accuracy metrics."""
    if df.empty or 'Voltage_V' not in df.columns: return {}
    
    results = {}
    for sensor in df['Sensor_Name'].unique():
        sub = df[df['Sensor_Name'] == sensor].dropna()
        if sub.empty: continue
        
        # Hardware Accuracy
        exp = (sub['Voltage_V'] / V_REF * ADC_MAX).astype(int)
        hw = (abs(sub['ADC_Value'] - exp) <= 1).mean() * 100
        results[sensor] = {'count': len(sub), 'hw': hw, 'ai': 0.0}

    # AI Accuracy
    clean = df.dropna(subset=['Voltage_V', 'Sensor_Name'])
    if clean['Sensor_Name'].nunique() >= 2 and len(clean) > 50:
        try:
            X = clean[['Voltage_V']]
            y = clean['Sensor_Name']
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            clf.fit(X, y)
            report = classification_report(y, clf.predict(X), output_dict=True, zero_division=0)
            for s in results:
                if s in report: results[s]['ai'] = report[s]['recall'] * 100
        except: pass
        
    return results

# --- 3. App State & Sidebar ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_master()

master_df = st.session_state.master_df

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2906/2906274.png", width=60)
    st.title("Control Center")
    st.markdown("---")
    
    st.subheader("1. Data Ingestion")
    if st.button("⚡ Initialize 3-Sensor Data", help="Generates Temp, Moisture, Light"):
        with st.spinner("Generating 3 Sensors..."):
            new_data = generate_spec_data()
            new_data.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_data
            time.sleep(0.5)
            st.rerun()
            
    uploaded = st.file_uploader("Or Upload CSV", type=['csv'])
    if uploaded:
        new_raw = pd.read_csv(uploaded)
        merged, msg = process_merge_fast(master_df, new_raw)
        if merged is not None:
            merged.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = merged
            st.success("Sync Complete")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error(msg)
            
    st.markdown("---")
    st.subheader("2. System Actions")
    if st.button("⚠️ Factory Reset", type="primary"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
        st.rerun()

# --- 4. Main Dashboard UI ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("SensorEdge Analytics")
    st.caption("3-Sensor Edge Device Monitoring")

with col_head2:
    status = "Online" if not master_df.empty else "Offline"
    color = "green" if not master_df.empty else "gray"
    st.markdown(f"""
        <div style="text-align: right; padding: 10px;">
            <span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 14px;">
                ● System {status}
            </span>
        </div>
    """, unsafe_allow_html=True)

tab_dash, tab_data = st.tabs(["📊 Executive Dashboard", "📋 Data Inspector"])

with tab_dash:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_recs = len(master_df)
    unique_sensors = master_df['Sensor_Name'].nunique() if not master_df.empty else 0
    
    kpi1.metric("Total Records", f"{total_recs:,}")
    kpi2.metric("Active Sensors", unique_sensors)
    
    analytics = get_analytics(master_df)
    avg_hw = np.mean([v['hw'] for v in analytics.values()]) if analytics else 0
    avg_ai = np.mean([v['ai'] for v in analytics.values()]) if analytics else 0
    
    kpi3.metric("Avg. Hardware Fidelity", f"{avg_hw:.1f}%")
    kpi4.metric("Avg. AI Confidence", f"{avg_ai:.1f}%")
    
    st.divider()
    
    # --- TABLE SECTION (Replaces Graph) ---
    st.markdown("### 🎯 Sensor Accuracy Matrix")
    if not master_df.empty and analytics:
        rows = []
        for s, m in analytics.items():
            rows.append({
                "Sensor Name": s, 
                "Record Count": f"{m['count']:,}",
                "Hardware Accuracy (ADC)": m['hw']/100, 
                "AI Model Confidence": m['ai']/100
            })
        
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sensor Name": st.column_config.TextColumn("Sensor Type", help="Source Sensor"),
                "Record Count": st.column_config.TextColumn("Samples"),
                "Hardware Accuracy (ADC)": st.column_config.ProgressColumn(
                    "Hardware Check", format="%.1f%%", min_value=0, max_value=1
                ),
                "AI Model Confidence": st.column_config.ProgressColumn(
                    "AI Prediction", format="%.1f%%", min_value=0, max_value=1
                )
            }
        )
    else:
        st.info("System is offline. Please initialize 3-Sensor data from the sidebar.")

with tab_data:
    st.markdown("### 🗃️ Master Data Ledger")
    
    if not master_df.empty:
        c1, c2 = st.columns([3, 1])
        with c1:
            filter_opt = st.radio("Status Filter:", ["All", "New (Green)", "Overlap (Red)", "Historical"], horizontal=True)
        with c2:
            st.download_button("📥 Export CSV", master_df.to_csv(index=False).encode('utf-8'), "sensor_data.csv", "text/csv")
            
        view_df = master_df.copy()
        if filter_opt == "New (Green)": view_df = view_df[view_df['Status'] == 'New']
        elif filter_opt == "Overlap (Red)": view_df = view_df[view_df['Status'] == 'Overlap']
        elif filter_opt == "Historical": view_df = view_df[view_df['Status'] == 'Historical']
        
        def row_styler(row):
            if row['Status'] == 'New': return ['background-color: #d1e7dd; color: #0f5132'] * len(row)
            if row['Status'] == 'Overlap': return ['background-color: #f8d7da; color: #842029'] * len(row)
            return [''] * len(row)
            
        try:
            st.dataframe(view_df.style.apply(row_styler, axis=1), use_container_width=True, height=500)
        except:
            st.dataframe(view_df, use_container_width=True)
            
    else:
        st.warning("No data found.")
