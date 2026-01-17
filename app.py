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
    """Forces all variations (node_1, light, LIGHT) to standard 3 names."""
    n = str(name).strip().lower()
    if 'temp' in n or 'node_1' in n or 'node 1' in n: return 'Temperature'
    if 'moist' in n or 'node_2' in n or 'node 2' in n: return 'Moisture'
    if 'light' in n or 'node_3' in n or 'node 3' in n: return 'Light'
    return n.title() # Fallback for unknown sensors

@st.cache_data(ttl=60)
def load_and_clean_master():
    """Loads data AND performs a deep clean to fix '6 sensor' bug."""
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            
            # 1. Standardize Columns
            df.columns = df.columns.str.strip().str.title()
            rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 
                          'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 'Adc': 'ADC_Value'}
            df.rename(columns=rename_map, inplace=True)
            
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.round('1s')
                
            # 2. FORCE FIX SENSOR NAMES (The Self-Healing Step)
            if 'Sensor_Name' in df.columns:
                df['Sensor_Name'] = df['Sensor_Name'].apply(normalize_sensor_names)
                
            # 3. FORCE REMOVE DUPLICATES (Clean up old messes)
            df = df.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
            
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

def process_merge_fast(master_df, new_df):
    """Turbo Merge: Optimizes for 100k+ rows."""
    
    # 1. Clean Column Names
    new_df.columns = new_df.columns.str.strip().str.title()
    rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 
                  'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 'Adc': 'ADC_Value',
                  'Node': 'Sensor_Name', 'Source': 'Sensor_Name'}
    new_df.rename(columns=rename_map, inplace=True)
    
    if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
        return None, "❌ Invalid Format: Missing Timestamp or Sensor_Name"

    # 2. Vectorized Processing
    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp']).dt.round('1s')
    if not master_df.empty:
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp']).dt.round('1s')

    # 3. Apply Name Mapping (Fixes 'light' vs 'Light')
    new_df['Sensor_Name'] = new_df['Sensor_Name'].apply(normalize_sensor_names)
    
    # 4. Fast Fill & Fix 0% Accuracy
    if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
    if 'ADC_Value' not in new_df.columns: new_df['ADC_Value'] = 0
    
    # Auto-calculate missing values
    mask_v_missing = (new_df['Voltage_V'] == 0) & (new_df['ADC_Value'] > 0)
    new_df.loc[mask_v_missing, 'Voltage_V'] = (new_df.loc[mask_v_missing, 'ADC_Value'] / ADC_MAX * V_REF).round(4)
    
    mask_a_missing = (new_df['ADC_Value'] == 0) & (new_df['Voltage_V'] > 0)
    new_df.loc[mask_a_missing, 'ADC_Value'] = (new_df.loc[mask_a_missing, 'Voltage_V'] / V_REF * ADC_MAX).astype(int)

    # 5. Fast Status Update
    new_df['Status'] = 'New'
    if not master_df.empty:
        master_df['Status'] = 'Historical'
        # Set Index for instantaneous lookup
        m_idx = master_df.set_index(['Timestamp', 'Sensor_Name']).index
        n_idx = new_df.set_index(['Timestamp', 'Sensor_Name']).index
        master_df.loc[master_df.set_index(['Timestamp', 'Sensor_Name']).index.isin(n_idx), 'Status'] = 'Overlap'

    # 6. Concat & Deduplicate
    combined = pd.concat([master_df, new_df])
    combined = combined.sort_values('Timestamp')
    
    # Drop duplicates (Keep first = keeps Master/Red rows)
    before_len = len(combined)
    final = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    removed = before_len - len(final)
    
    return final, f"Merged! Removed {removed} duplicates."

def get_analytics(df):
    """Calculates accuracy on a sample to stay fast."""
    if df.empty or 'Voltage_V' not in df.columns: return {}
    
    results = {}
    groups = df.groupby('Sensor_Name')
    
    for sensor, sub in groups:
        count = len(sub)
        exp = (sub['Voltage_V'] / V_REF * ADC_MAX).astype(int)
        hw = (np.abs(sub['ADC_Value'] - exp) <= 1).mean() * 100
        results[sensor] = {'count': count, 'hw': hw, 'ai': 0.0}

    # AI Check (Limit training data for speed)
    clean = df.dropna(subset=['Voltage_V', 'Sensor_Name'])
    if clean['Sensor_Name'].nunique() >= 2:
        sample_size = min(1000, len(clean))
        clean_sample = clean.sample(sample_size, random_state=42)
        try:
            X = clean_sample[['Voltage_V']]
            y = clean_sample['Sensor_Name']
            clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            clf.fit(X, y)
            preds = clf.predict(X)
            report = classification_report(y, preds, output_dict=True, zero_division=0)
            for s in results:
                if s in report: results[s]['ai'] = report[s]['recall'] * 100
        except: pass
        
    return results

# --- 3. App State ---
# This line now calls the CLEANING function instead of just loading
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_and_clean_master()

master_df = st.session_state.master_df

# --- Sidebar ---
with st.sidebar:
    st.title("⚡ Speed Control")
    st.markdown("---")
    
    if st.button("Initialize 3-Sensor Data (Fast)"):
        with st.spinner("Generating..."):
            new_data = generate_spec_data()
            new_data.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_data
            time.sleep(0.5)
            st.rerun()
            
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        new_raw = pd.read_csv(uploaded)
        merged, msg = process_merge_fast(master_df, new_raw)
        
        if merged is not None:
            merged.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = merged
            st.success(msg)
            time.sleep(0.5)
            st.rerun()
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
    # Top Stats
    k1, k2, k3, k4 = st.columns(4)
    analytics = get_analytics(master_df)
    
    if analytics:
        avg_hw = np.mean([v['hw'] for v in analytics.values()])
        avg_ai = np.mean([v['ai'] for v in analytics.values()])
    else:
        avg_hw, avg_ai = 0.0, 0.0
        
    k1.metric("Total Records", f"{len(master_df):,}")
    k2.metric("Active Sensors", master_df['Sensor_Name'].nunique())
    k3.metric("Hardware Fidelity", f"{avg_hw:.1f}%")
    k4.metric("AI Confidence", f"{avg_ai:.1f}%")
    
    st.divider()
    
    # TABS
    tab1, tab2 = st.tabs(["🎯 Accuracy Matrix", "🔍 Data Inspector (Fast View)"])
    
    with tab1:
        if analytics:
            rows = [{"Sensor": s, "Count": f"{m['count']:,}", "HW Accuracy": m['hw']/100, "AI Accuracy": m['ai']/100} for s, m in analytics.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                column_config={
                    "HW Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                    "AI Accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1)
                })
        else:
            st.info("No analytics data.")
            
    with tab2:
        st.caption(f"Showing top 1,000 rows of {len(master_df):,} for performance. Download full file below.")
        
        view_df = master_df.head(1000).copy()
        
        def highlight_status(val):
            if val == 'New': return 'background-color: #d1e7dd; color: #0f5132' # Green
            if val == 'Overlap': return 'background-color: #f8d7da; color: #842029' # Red
            return ''
            
        try:
            st.dataframe(view_df.style.map(highlight_status, subset=['Status']), use_container_width=True)
        except:
            st.dataframe(view_df, use_container_width=True)
            
        st.download_button("📥 Download Full Master CSV", master_df.to_csv(index=False).encode('utf-8'), "master_data.csv", "text/csv")

else:
    st.info("System Offline. Use sidebar to load data.")
