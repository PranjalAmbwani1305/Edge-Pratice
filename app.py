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

st.set_page_config(page_title="Sensor Analytics Pro", layout="wide")
st.title("⚡ Sensor Manager & Per-Sensor Analytics")

# --- 1. Data Generation (Spec-Compliant) ---
def generate_demo_data():
    """Generates 24-hour dataset based on exact sensor specs."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    duration_sec = 24 * 3600  # 24 Hours
    
    # Helper to generate individual sensor data
    def make_sensor(name, interval_sec, v_min, v_max):
        # Create time range
        times = [start_time + timedelta(seconds=i) for i in range(0, duration_sec, interval_sec)]
        
        # Generate Voltages
        volts = np.random.uniform(v_min, v_max, len(times))
        
        # Calculate ADC (10-bit resolution)
        adcs = (volts / V_REF * ADC_MAX).astype(int)
        
        return pd.DataFrame({
            'Timestamp': times, 
            'Sensor_Name': name, 
            'Voltage_V': np.round(volts, 4), 
            'ADC_Value': adcs, 
            'Status': 'Historical' # Default to white for master data
        })
    
    # 1. Temperature: Every 2s, 2V-4V
    df_temp = make_sensor("Temperature", 2, 2.0, 4.0)
    
    # 2. Moisture: Every 4h (14400s), 1.2V-3V
    df_moist = make_sensor("Moisture", 14400, 1.2, 3.0)
    
    # 3. Light: Every 5s, 0V-5V
    df_light = make_sensor("Light", 5, 0.0, 5.0)
    
    # Combine & Sort
    master = pd.concat([df_temp, df_moist, df_light])
    master = master.sort_values('Timestamp').reset_index(drop=True)
    
    return master

# --- 2. Optimized Processing Functions ---
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

def process_merge_fast(master_df, new_df):
    """Vectorized Merge Strategy (Fast & Correct)."""
    # Standardize New Data
    new_df.columns = new_df.columns.str.strip().str.title()
    rename_map = {
        'Time': 'Timestamp', 'Date': 'Timestamp', 
        'Sensor': 'Sensor_Name', 'Name': 'Sensor_Name', 
        'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 
        'Adc': 'ADC_Value'
    }
    new_df.rename(columns=rename_map, inplace=True)
    
    # Validation
    if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
        return None, "Missing columns: Timestamp, Sensor_Name"

    # Set Status
    new_df['Status'] = 'New'
    
    if not master_df.empty:
        master_df['Status'] = 'Historical'
        
        # Fast Overlap Check
        master_idx = pd.MultiIndex.from_frame(master_df[['Timestamp', 'Sensor_Name']])
        new_idx = pd.MultiIndex.from_frame(new_df[['Timestamp', 'Sensor_Name']])
        overlap_mask = master_idx.isin(new_idx)
        master_df.loc[overlap_mask, 'Status'] = 'Overlap'

    # Concatenate & Deduplicate
    combined = pd.concat([master_df, new_df])
    combined['Timestamp'] = pd.to_datetime(combined['Timestamp'])
    combined = combined.sort_values('Timestamp')
    
    # Keep Master (Red/White) over New (Green)
    final_df = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
    
    return final_df, "Success"

def calculate_per_sensor_accuracy(df):
    """Calculates Hardware & AI Accuracy for EACH sensor type."""
    if df.empty or 'Voltage_V' not in df.columns:
        return {}

    results = {}
    sensors = df['Sensor_Name'].unique()
    
    # Hardware Accuracy
    for sensor in sensors:
        s_df = df[df['Sensor_Name'] == sensor].dropna(subset=['Voltage_V', 'ADC_Value'])
        if not s_df.empty:
            expected = (s_df['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            matches = (abs(s_df['ADC_Value'] - expected) <= 1)
            hw_acc = matches.mean() * 100
            results[sensor] = {'hw': hw_acc, 'ai': 0.0, 'count': len(s_df)}
        else:
            results[sensor] = {'hw': 0.0, 'ai': 0.0, 'count': 0}

    # AI Model Accuracy
    df_clean = df.dropna(subset=['Voltage_V', 'Sensor_Name'])
    if df_clean['Sensor_Name'].nunique() >= 2 and len(df_clean) > 20:
        try:
            X = df_clean[['Voltage_V']]
            y = df_clean['Sensor_Name']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            clf = RandomForestClassifier(n_estimators=20, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for sensor in sensors:
                if sensor in report:
                    results[sensor]['ai'] = report[sensor]['recall'] * 100
        except:
            pass
            
    return results

# --- 3. State Management ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_master()

master_df = st.session_state.master_df

# --- 4. Sidebar Actions ---
with st.sidebar:
    st.header("Actions")
    
    # A. Generate Data (Restored Feature)
    if st.button("Initialize Demo Data (24h)"):
        with st.spinner("Generating 24-hour baseline..."):
            new_master = generate_demo_data()
            new_master.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_master
            st.success("Generated 60k+ records!")
            time.sleep(1)
            st.rerun()

    st.divider()

    # B. Upload
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        new_df_raw = pd.read_csv(uploaded_file)
        with st.spinner("Processing..."):
            merged_df, msg = process_merge_fast(master_df, new_df_raw)
            if merged_df is not None:
                merged_df.to_csv(MASTER_CSV, index=False)
                st.session_state.master_df = merged_df
                st.success("Merged Successfully!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(msg)

    st.divider()
    if st.button("Reset Data"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
        st.rerun()

# --- 5. Main Dashboard ---

# A. Global Stats
col1, col2 = st.columns(2)
col1.metric("Total Records", len(master_df))
col2.metric("Unique Sensors", master_df['Sensor_Name'].nunique() if not master_df.empty else 0)

st.divider()

# B. Sensor-Wise Accuracy Table
st.subheader("🎯 Sensor-Wise Accuracy Breakdown")

if not master_df.empty:
    acc_data = calculate_per_sensor_accuracy(master_df)
    rows = []
    for sensor, metrics in acc_data.items():
        rows.append({
            "Sensor Name": sensor,
            "Record Count": metrics['count'],
            "Hardware Accuracy (ADC)": f"{metrics['hw']:.1f}%",
            "AI Recognition Accuracy": f"{metrics['ai']:.1f}%"
        })
    
    acc_df = pd.DataFrame(rows)
    st.dataframe(
        acc_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Hardware Accuracy (ADC)": st.column_config.ProgressColumn(
                "Hardware Fidelity", format="%s", min_value=0, max_value=100
            ),
            "AI Recognition Accuracy": st.column_config.ProgressColumn(
                "Model Confidence", format="%s", min_value=0, max_value=100
            )
        }
    )
else:
    st.info("No data available. Click 'Initialize Demo Data' in sidebar to start.")

st.divider()

# C. Master Data View
st.subheader("📋 Master Data Ledger")

def color_status(val):
    if val == 'New': return 'background-color: #90EE90; color: black'
    elif val == 'Overlap': return 'background-color: #FF7F7F; color: black'
    return ''

if not master_df.empty:
    try:
        st.dataframe(master_df.style.map(color_status, subset=['Status']), use_container_width=True)
    except AttributeError:
        try:
             st.dataframe(master_df.style.applymap(color_status, subset=['Status']), use_container_width=True)
        except Exception:
             st.dataframe(master_df, use_container_width=True)
    except Exception:
        st.dataframe(master_df, use_container_width=True)
    
    st.download_button(
        "Download Master CSV", 
        master_df.to_csv(index=False).encode('utf-8'), 
        "master_dataset.csv", 
        "text/csv"
    )
