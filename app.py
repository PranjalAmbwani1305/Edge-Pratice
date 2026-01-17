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

# --- 1. Page Config & CSS ---
st.set_page_config(
    page_title="SensorEdge Enterprise", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="expanded"
)

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

# --- 2. Core Logic ---
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
    
    return pd.concat([
        make("Temperature", 2, 2.0, 4.0),
        make("Moisture", 14400, 1.2, 3.0),
        make("Light", 5, 0.0, 5.0)
    ]).sort_values('Timestamp').reset_index(drop=True)

def normalize_sensor_names(name):
    """Forces incoming names to match the 3 Master Sensors."""
    name = str(name).strip().lower()
    if 'temp' in name or 'node_1' in name or 'node 1' in name:
        return 'Temperature'
    if 'moist' in name or 'node_2' in name or 'node 2' in name:
        return 'Moisture'
    if 'light' in name or 'node_3' in name or 'node 3' in name:
        return 'Light'
    return name.title() # Fallback

def process_merge_fast(master_df, new_df):
    """Smart Merge: Normalizes Names, Rounds Time, and Deduplicates."""
    
    # 1. Clean Column Names
    new_df.columns = new_df.columns.str.strip().str.title()
    rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 
                  'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 'Adc': 'ADC_Value',
                  'Node': 'Sensor_Name', 'Source': 'Sensor_Name'}
    new_df.rename(columns=rename_map, inplace=True)
    
    if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
        return None, "❌ Invalid Format: Missing Timestamp or Sensor_Name"

    # 2. FORCE TIMESTAMP FORMAT (Round to nearest second to catch duplicates)
    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp']).dt.round('1s')
    if not master_df.empty:
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp']).dt.round('1s')

    # 3. FORCE SENSOR NAME MAPPING (The Fix)
    new_df['Sensor_Name'] = new_df['Sensor_Name'].apply(normalize_sensor_names)
    
    # 4. Auto-Calculate Missing Voltage/ADC
    if 'Voltage_V' not in new_df.columns and 'ADC_Value' in new_df.columns:
        new_df['Voltage_V'] = (new_df['ADC_Value'] / ADC_MAX * V_REF).round(4)
    if 'ADC_Value' not in new_df.columns and 'Voltage_V' in new_df.columns:
        new_df['ADC_Value'] = (new_df['Voltage_V'] / V_REF * ADC_MAX).astype(int)
    
    # Fill NaN defaults
    if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
    if 'ADC_Value' not in new_df.columns: new_df['ADC_Value'] = 0

    # 5. Set Status & Detect Overlap
    new_df['Status'] = 'New'
    
    if not master_df.empty:
        master_df['Status'] = 'Historical'
        
        # Create keys for fast lookup
        m_idx = pd.MultiIndex.from_frame(master_df[['Timestamp', 'Sensor_Name']])
        n_idx = pd.MultiIndex.from_frame(new_df[['Timestamp', 'Sensor_Name']])
        
        # Mark Overlaps in Master
        overlap_mask = m_idx.isin(n_idx)
        master_df.loc[overlap_mask, 'Status'] = 'Overlap'

    # 6. CONCAT & REMOVE DUPLICATES
    # We put Master FIRST so that if duplicates exist, 'keep=first' keeps the Master (Red/White) row
    combined = pd.concat([master_df, new_df])
    combined = combined.sort_values('Timestamp')
    
    # STRICT DEDUPLICATION
    before_len = len(combined)
    final = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    after_len = len(final)
    
    removed_count = before_len - after_len
    
    return final, f"Merged! Removed {removed_count} duplicate rows."

def get_analytics(df):
    """Calculates granular accuracy metrics."""
    if df.empty or 'Voltage_V' not in df.columns: return {}
    
    results = {}
    for sensor in df['Sensor_Name'].unique():
        sub = df[df['Sensor_Name'] == sensor].dropna()
        if sub.empty: continue
        
        # Skip if voltage is missing/zero
        if sub['Voltage_V'].sum() == 0 and sub['ADC_Value'].sum() > 0:
             sub['Voltage_V'] = (sub['ADC_Value'] / ADC_MAX * V_REF)

        exp = (sub['Voltage_V'] / V_REF * ADC_MAX).astype(int)
        hw = (abs(sub['ADC_Value'] - exp) <= 1).mean() * 100
        
        results[sensor] = {'count': len(sub), 'hw': hw, 'ai': 0.0}

    # AI Accuracy
    clean = df.dropna(subset=['Voltage_V', 'Sensor_Name'])
    if clean['Sensor_Name'].nunique() >= 2 and len(clean) > 50 and clean['Voltage_V'].std() > 0.1:
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

# --- 3. App State ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_master()

master_df = st.session_state.master_df

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2906/2906274.png", width=60)
    st.title("Control Center")
    st.markdown("---")
    
    if st.button("⚡ Initialize 3-Sensor Data"):
        with st.spinner("Generating..."):
            new_data = generate_spec_data()
            new_data.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_data
            time.sleep(0.5)
            st.rerun()
            
    uploaded = st.file_uploader("Or Upload CSV", type=['csv'])
    if uploaded:
        new_raw = pd.read_csv(uploaded)
        # Perform Merge
        merged, msg = process_merge_fast(master_df, new_raw)
        
        if merged is not None:
            # Force Save
            merged.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = merged
            st.success(msg) # Show duplicate removal count
            time.sleep(1)
            st.rerun()
        else:
            st.error(msg)
            
    st.markdown("---")
    if st.button("⚠️ Factory Reset", type="primary"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
        st.rerun()

# --- 4. Main Dashboard ---
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
    
    analytics = get_analytics(master_df)
    
    if analytics:
        avg_hw = np.mean([v['hw'] for v in analytics.values()])
        avg_ai = np.mean([v['ai'] for v in analytics.values()])
    else:
        avg_hw, avg_ai = 0.0, 0.0
    
    kpi1.metric("Total Records", f"{total_recs:,}")
    kpi2.metric("Active Sensors", unique_sensors)
    kpi3.metric("Avg. Hardware Fidelity", f"{avg_hw:.1f}%")
    kpi4.metric("Avg. AI Confidence", f"{avg_ai:.1f}%")
    
    st.divider()
    
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
                "Sensor Name": st.column_config.TextColumn("Sensor Type"),
                "Record Count": st.column_config.TextColumn("Samples"),
                "Hardware Accuracy (ADC)": st.column_config.ProgressColumn(
                    "Hardware Check", format="%.1f%%", min_value=0, max_value=1
                ),
                "AI Model Confidence": st.column_config.ProgressColumn(
                    "AI Prediction", format="%.1f%%", min_value=0, max_value=1
                )
            }
        )
    elif not master_df.empty and not analytics:
        st.warning("Data loaded, but metrics unavailable. Check if Voltage/ADC columns are valid.")
    else:
        st.info("System is offline. Please initialize data from the sidebar.")

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
