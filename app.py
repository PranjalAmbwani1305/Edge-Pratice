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

# --- 1. Page Config ---
st.set_page_config(
    page_title="SensorEdge", 
    layout="wide", 
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
def normalize_sensor_names(name):
    """Forces all variations to standard 3 names."""
    n = str(name).strip().lower()
    if 'temp' in n or 'node_1' in n or 'node 1' in n: return 'Temperature'
    if 'moist' in n or 'node_2' in n or 'node 2' in n: return 'Moisture'
    if 'light' in n or 'node_3' in n or 'node 3' in n: return 'Light'
    return n.title() 

@st.cache_data(ttl=60)
def load_and_clean_master():
    """Loads and sanitizes the master dataset."""
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            # 1. Clean Headers
            df.columns = df.columns.str.strip().str.title()
            rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 
                          'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 'Adc': 'ADC_Value'}
            df.rename(columns=rename_map, inplace=True)
            df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate cols

            # 2. Add Missing Columns
            for col in ['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status']:
                if col not in df.columns:
                    df[col] = 0 if 'Value' in col or 'Voltage' in col else None
            
            # 3. Format Data
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.round('1s')
            if 'Sensor_Name' in df.columns:
                df['Sensor_Name'] = df['Sensor_Name'].apply(normalize_sensor_names)
                
            return df.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
        except:
            pass
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])

def generate_spec_data():
    """Generates 24H Golden Dataset (00:00 to 23:59)."""
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

def process_merge_fast(master_df, new_df):
    """Merges data and returns detailed stats."""
    
    # --- 1. Pre-Processing ---
    new_df.columns = new_df.columns.str.strip().str.title()
    rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 
                  'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 'Adc': 'ADC_Value',
                  'Node': 'Sensor_Name', 'Source': 'Sensor_Name'}
    new_df.rename(columns=rename_map, inplace=True)
    new_df = new_df.loc[:, ~new_df.columns.duplicated()]

    if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
        return None, "❌ Error: Missing 'Timestamp' or 'Sensor_Name' columns."

    # --- 2. Normalize Incoming Data ---
    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp']).dt.round('1s')
    new_df['Sensor_Name'] = new_df['Sensor_Name'].apply(normalize_sensor_names)
    
    # Auto-Repair Missing Values
    if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
    if 'ADC_Value' not in new_df.columns: new_df['ADC_Value'] = 0
    
    mask_v_miss = (new_df['Voltage_V'] == 0) & (new_df['ADC_Value'] > 0)
    new_df.loc[mask_v_miss, 'Voltage_V'] = (new_df.loc[mask_v_miss, 'ADC_Value'] / ADC_MAX * V_REF).round(4)
    
    mask_a_miss = (new_df['ADC_Value'] == 0) & (new_df['Voltage_V'] > 0)
    new_df.loc[mask_a_miss, 'ADC_Value'] = (new_df.loc[mask_a_miss, 'Voltage_V'] / V_REF * ADC_MAX).astype(int)

    # --- 3. Identify Status BEFORE Merge ---
    new_df['Status'] = 'New' # Default to Green
    
    if not master_df.empty:
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp']).dt.round('1s')
        master_df['Status'] = 'Historical' # Reset Master to White
        
        # Check Overlaps
        m_idx = master_df.set_index(['Timestamp', 'Sensor_Name']).index
        n_idx = new_df.set_index(['Timestamp', 'Sensor_Name']).index
        
        # Mark Overlapping Master Rows as RED
        overlap_mask = m_idx.isin(n_idx)
        master_df.loc[overlap_mask, 'Status'] = 'Overlap'

    # --- 4. Merge & Deduplicate ---
    initial_master_count = len(master_df)
    incoming_count = len(new_df)
    
    combined = pd.concat([master_df, new_df]).sort_values('Timestamp')
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # Keep 'first' preserves the Master row (which is now marked Red/Overlap or White/Historical)
    final = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    
    final_count = len(final)
    added_rows = final_count - initial_master_count
    merged_duplicates = incoming_count - added_rows
    
    report = f"""
    **Merge Report:**
    * 📥 **Incoming Rows:** {incoming_count:,}
    * ✨ **New Rows Added:** {added_rows:,} (Green)
    * 🔄 **Duplicates Merged:** {merged_duplicates:,} (Red/Overlap)
    """
    
    return final, report

def get_analytics(df):
    """Calculates accuracy safely."""
    if df.empty or 'Voltage_V' not in df.columns: return {}
    results = {}
    for sensor, sub in df.groupby('Sensor_Name'):
        exp = (sub['Voltage_V'] / V_REF * ADC_MAX).astype(int)
        hw = (np.abs(sub['ADC_Value'] - exp) <= 1).mean() * 100
        results[sensor] = {'count': len(sub), 'hw': hw, 'ai': 0.0}
    
    # AI Check (Sampled)
    clean = df.dropna(subset=['Voltage_V', 'Sensor_Name'])
    if clean['Sensor_Name'].nunique() >= 2:
        try:
            s_df = clean.sample(min(1000, len(clean)), random_state=42)
            clf = RandomForestClassifier(n_estimators=10, max_depth=5).fit(s_df[['Voltage_V']], s_df['Sensor_Name'])
            report = classification_report(s_df['Sensor_Name'], clf.predict(s_df[['Voltage_V']]), output_dict=True, zero_division=0)
            for s in results:
                if s in report: results[s]['ai'] = report[s]['recall'] * 100
        except: pass
    return results

# --- 3. App State ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_and_clean_master()

master_df = st.session_state.master_df

# --- Sidebar ---
with st.sidebar:
    st.title("⚡ Speed Control")
    st.markdown("---")
    
    if st.button("Initialize 3-Sensor Data (Fast)"):
        with st.spinner("Generating 24h Data..."):
            new_data = generate_spec_data()
            new_data.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_data
            time.sleep(0.5)
            st.rerun()
            
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        new_raw = pd.read_csv(uploaded)
        final_df, msg = process_merge_fast(master_df, new_raw)
        
        if final_df is not None:
            final_df.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = final_df
            st.success("Processed!")
            st.markdown(msg) # Show detailed report
        else:
            st.error(msg)
            
    st.markdown("---")
    if st.button("Reset System", type="primary"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
        st.rerun()

# --- 4. Dashboard ---
st.title("SensorEdge Pro ⚡")

if not master_df.empty:
    k1, k2, k3, k4 = st.columns(4)
    analytics = get_analytics(master_df)
    
    avg_hw = np.mean([v['hw'] for v in analytics.values()]) if analytics else 0.0
    avg_ai = np.mean([v['ai'] for v in analytics.values()]) if analytics else 0.0
        
    k1.metric("Total Records", f"{len(master_df):,}")
    k2.metric("Active Sensors", master_df['Sensor_Name'].nunique())
    k3.metric("Hardware Fidelity", f"{avg_hw:.1f}%")
    k4.metric("AI Confidence", f"{avg_ai:.1f}%")
    
    st.divider()
    
    tab1, tab2 = st.tabs(["🎯 Accuracy Matrix", "🔍 Data Inspector"])
    
    with tab1:
        if analytics:
            rows = [{"Sensor": s, "Count": f"{m['count']:,}", "HW Accuracy": m['hw']/100, "AI Accuracy": m['ai']/100} for s, m in analytics.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                column_config={
                    "HW Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                    "AI Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1)
                })
        else:
            st.info("No valid data for analytics.")
            
    with tab2:
        st.caption(f"Showing sample of {len(master_df):,} rows. Overlaps are RED, New data is GREEN.")
        
        # Robust Styler
        def highlight_status(val):
            if val == 'New': return 'background-color: #d1e7dd; color: #0f5132'
            if val == 'Overlap': return 'background-color: #f8d7da; color: #842029'
            return ''
        
        view = master_df.head(1000).copy()
        try:
            st.dataframe(view.style.map(highlight_status, subset=['Status']), use_container_width=True)
        except:
            st.dataframe(view, use_container_width=True)
            
        st.download_button("📥 Download Master CSV", master_df.to_csv(index=False).encode('utf-8'), "master_data.csv", "text/csv")

else:
    st.info("System Offline. Use sidebar to Initialize Data.")
