import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta, date
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Configuration ---
MASTER_CSV = 'master_edge_sensor_data.csv'
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

# --- 2. Logic: Generation ---
def generate_sensor_data(sensor_name, interval_seconds, v_min, v_max, duration_hours, start_time):
    #
    total_seconds = duration_hours * 3600
    timestamps = [start_time + timedelta(seconds=i) for i in range(0, total_seconds, interval_seconds)]
    
    data = []
    for ts in timestamps:
        voltage = np.random.uniform(v_min, v_max)
        adc_value = int((voltage / V_REF) * ADC_MAX) #
        
        data.append({
            "Timestamp": ts,
            "Sensor_Name": sensor_name,
            "Voltage_V": round(voltage, 2),
            "ADC_Value": adc_value,
            "Status": "Historical"
        })
    return pd.DataFrame(data)

def generate_full_dataset(start_date_obj):
    #
    start_time = datetime.combine(start_date_obj, datetime.min.time())
    duration = 24 
    
    df_temp = generate_sensor_data("Temperature", 2, 2.0, 4.0, duration, start_time)
    df_moist = generate_sensor_data("Moisture", 14400, 1.2, 3.0, duration, start_time)
    df_light = generate_sensor_data("Light", 5, 0.0, 5.0, duration, start_time)
    
    full_df = pd.concat([df_temp, df_moist, df_light])
    full_df = full_df.sort_values(by="Timestamp").reset_index(drop=True)
    return full_df

# --- 3. Logic: Merge ---
def normalize_sensor_names(name):
    n = str(name).strip().lower()
    if 'temp' in n or 'node_1' in n: return 'Temperature'
    if 'moist' in n or 'node_2' in n: return 'Moisture'
    if 'light' in n or 'node_3' in n: return 'Light'
    return n.title()

def merge_logic_exact(master_df, new_df, filename="uploaded_file.csv"):
    log_buffer = []
    log_buffer.append(f"Loading Master: {MASTER_CSV}...")
    log_buffer.append(f"Loading New Data: {filename}...")
    
    # 1. Prepare New Data
    new_df.columns = new_df.columns.str.strip().str.title()
    rename_map = {
        'Time': 'Timestamp', 'Date': 'Timestamp', 
        'Sensor': 'Sensor_Name', 'Node': 'Sensor_Name',
        'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 
        'Adc': 'ADC_Value', 'Adc_Value': 'ADC_Value'
    }
    new_df.rename(columns=rename_map, inplace=True)
    new_df = new_df.loc[:, ~new_df.columns.duplicated()]

    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp']).dt.round('1s')
    new_df['Sensor_Name'] = new_df['Sensor_Name'].apply(normalize_sensor_names)
    
    if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
    if 'ADC_Value' not in new_df.columns: new_df['ADC_Value'] = 0
    
    new_df['Voltage_V'] = pd.to_numeric(new_df['Voltage_V'], errors='coerce').fillna(0.0)
    new_df['ADC_Value'] = pd.to_numeric(new_df['ADC_Value'], errors='coerce').fillna(0).astype(int)
    new_df['Status'] = 'New'

    # 2. Logic: Overlaps
    if not master_df.empty:
        master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp']).dt.round('1s')
        master_df['Status'] = 'Historical'
        m_idx = master_df.set_index(['Timestamp', 'Sensor_Name']).index
        n_idx = new_df.set_index(['Timestamp', 'Sensor_Name']).index
        master_df.loc[m_idx.isin(n_idx), 'Status'] = 'Overlap'

    # 3. Notebook Merge Logic
    log_buffer.append("Merging datasets...")
    
    old_count = len(master_df)
    new_records_count = len(new_df)
    
    combined = pd.concat([master_df, new_df])
    
    # keep='first' logic
    final_df = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    final_df = final_df.sort_values(by='Timestamp').reset_index(drop=True)
    
    final_count = len(final_df)
    added_count = final_count - old_count
    ignored_count = new_records_count - added_count
    
    # 4. Generate EXACT Log Output
    log_buffer.append("-" * 40)
    log_buffer.append("UPDATE SUCCESSFUL")
    log_buffer.append("-" * 40)
    log_buffer.append(f"Previous Master Size : {old_count}")
    log_buffer.append(f"New File Size        : {new_records_count}")
    log_buffer.append(f"Actually Added       : {added_count}")
    log_buffer.append(f"Duplicates Ignored   : {ignored_count}")
    log_buffer.append(f"New Master Size      : {final_count}")
    
    return final_df, "\n".join(log_buffer)

def get_analytics(df):
    if df.empty or 'Voltage_V' not in df.columns: return {}
    results = {}
    
    for sensor, sub_raw in df.groupby('Sensor_Name'):
        sub = sub_raw.copy()
        sub['Voltage_V'] = pd.to_numeric(sub['Voltage_V'], errors='coerce').fillna(0.0)
        sub['ADC_Value'] = pd.to_numeric(sub['ADC_Value'], errors='coerce').fillna(0)
        
        if len(sub) > 0:
            exp = (sub['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            hw = (np.abs(sub['ADC_Value'] - exp) <= 1).mean() * 100
        else:
            hw = 0.0
            
        results[sensor] = {'count': len(sub), 'hw': hw, 'ai': 0.0}
    
    # AI Check
    clean = df.dropna(subset=['Voltage_V', 'Sensor_Name']).copy()
    clean['Voltage_V'] = pd.to_numeric(clean['Voltage_V'], errors='coerce').fillna(0)
    
    if clean['Sensor_Name'].nunique() >= 2:
        try:
            s_df = clean.sample(min(1000, len(clean)), random_state=42)
            clf = RandomForestClassifier(n_estimators=10, max_depth=5).fit(s_df[['Voltage_V']], s_df['Sensor_Name'])
            report = classification_report(s_df['Sensor_Name'], clf.predict(s_df[['Voltage_V']]), output_dict=True, zero_division=0)
            for s in results:
                if s in report: results[s]['ai'] = report[s]['recall'] * 100
        except: pass
    return results

# --- 4. Main App ---

# Initialize State
if 'master_df' not in st.session_state:
    if os.path.exists(MASTER_CSV):
        try:
            st.session_state.master_df = pd.read_csv(MASTER_CSV)
            st.session_state.master_df['Timestamp'] = pd.to_datetime(st.session_state.master_df['Timestamp'])
            st.session_state.master_df['Voltage_V'] = pd.to_numeric(st.session_state.master_df['Voltage_V'], errors='coerce').fillna(0.0)
            st.session_state.master_df['ADC_Value'] = pd.to_numeric(st.session_state.master_df['ADC_Value'], errors='coerce').fillna(0).astype(int)
        except:
            st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])
    else:
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])

# Initialize Merge Log State
if 'merge_log' not in st.session_state:
    st.session_state.merge_log = None

master_df = st.session_state.master_df

# Sidebar
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("1. Initialize")
    if st.button("Generate & Reset"):
        with st.spinner("Generating..."):
            start_date = datetime.now().date()
            new_data = generate_full_dataset(start_date)
            new_data.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_data
            st.session_state.merge_log = None # Clear old log
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.subheader("2. Merge")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded:
        if st.button("Run Merge Script"):
            new_raw = pd.read_csv(uploaded)
            final_df, log_msg = merge_logic_exact(master_df, new_raw, uploaded.name)
            
            final_df.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = final_df
            st.session_state.merge_log = log_msg # Save log to state
            
            # Force Rerun to update main page
            st.rerun()

    st.markdown("---")
    if st.button("Factory Reset"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])
        st.session_state.merge_log = None
        st.rerun()

# --- Main Dashboard ---
st.title("SensorEdge")

# --- Display Merge Log Here (Top of Main Page) ---
if st.session_state.merge_log:
    st.success("Merge Completed Successfully")
    st.code(st.session_state.merge_log, language="text")
    if st.button("Clear Log"):
        st.session_state.merge_log = None
        st.rerun()
    st.divider()

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
    
    tab1, tab2 = st.tabs(["Accuracy Matrix", "Data Inspector"])
    
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
        st.caption(f"Showing sample of {len(master_df):,} rows.")
        
        def highlight_status(val):
            if val == 'New': return 'background-color: #d1e7dd; color: #0f5132'
            if val == 'Overlap': return 'background-color: #f8d7da; color: #842029'
            return ''
        
        view = master_df.head(1000).copy()
        try:
            st.dataframe(view.style.map(highlight_status, subset=['Status']), use_container_width=True)
        except:
            st.dataframe(view, use_container_width=True)
            
        st.download_button("Download Master CSV", master_df.to_csv(index=False).encode('utf-8'), MASTER_CSV, "text/csv")
else:
    st.info("System Offline. Use sidebar to Initialize Data.")
