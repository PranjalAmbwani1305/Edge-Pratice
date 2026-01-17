import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta, date

# --- Configuration ---
MASTER_CSV = 'master_edge_sensor_data.csv'
V_REF = 5.0        
ADC_MAX = 1023     

# --- 1. Page Config ---
st.set_page_config(
    page_title="SensorEdge Enterprise", 
    layout="wide", 
    page_icon="⚡",
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

# --- 2. Logic from sensor_database.ipynb ---
def generate_sensor_data(sensor_name, interval_seconds, v_min, v_max, duration_hours, start_time):
    # Calculate total seconds
    total_seconds = duration_hours * 3600
    
    # Generate timestamps list
    timestamps = [start_time + timedelta(seconds=i) for i in range(0, total_seconds, interval_seconds)]
    
    # Generate Data
    data = []
    for ts in timestamps:
        voltage = np.random.uniform(v_min, v_max)
        adc_value = int((voltage / V_REF) * ADC_MAX)
        
        data.append({
            "Timestamp": ts,
            "Sensor_Name": sensor_name,
            "Voltage_V": round(voltage, 2),
            "ADC_Value": adc_value
        })
    
    return pd.DataFrame(data)

def generate_full_dataset(start_date_obj):
    """Orchestrates the generation of all 3 sensors."""
    # Convert date object to datetime (Midnight)
    start_time = datetime.combine(start_date_obj, datetime.min.time())
    duration = 24 # Hours
    
    # 1. Temperature (Every 2s, 2V-4V)
    df_temp = generate_sensor_data("Temperature", 2, 2.0, 4.0, duration, start_time)
    
    # 2. Moisture (Every 4h, 1.2V-3V)
    df_moist = generate_sensor_data("Moisture", 14400, 1.2, 3.0, duration, start_time)
    
    # 3. Light (Every 5s, 0V-5V)
    df_light = generate_sensor_data("Light", 5, 0.0, 5.0, duration, start_time)
    
    # Combine & Sort
    full_df = pd.concat([df_temp, df_moist, df_light])
    full_df = full_df.sort_values(by="Timestamp").reset_index(drop=True)
    
    return full_df

# --- 3. Logic from Update_Master.ipynb ---
def merge_new_file_into_master(master_df, new_df):
    """
    Implements the exact logic from Update_Master.ipynb:
    Concat -> Drop Duplicates (subset=['Timestamp', 'Sensor_Name'], keep='first')
    """
    old_count = len(master_df)
    new_records_count = len(new_df)

    # 1. Ensure Timestamp format
    master_df['Timestamp'] = pd.to_datetime(master_df['Timestamp'])
    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'])

    # 2. Concat
    combined_df = pd.concat([master_df, new_df])

    # 3. Remove Duplicates
    # keep='first' preserves Master data if overlaps occur
    combined_df = combined_df.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')

    # 4. Sort
    combined_df = combined_df.sort_values(by='Timestamp').reset_index(drop=True)

    # Statistics
    final_count = len(combined_df)
    added_count = final_count - old_count
    ignored_count = new_records_count - added_count
    
    return combined_df, added_count, ignored_count

# --- 4. Helper: Load Master ---
def load_master():
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except:
            pass
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])

# --- 5. Streamlit App Layout ---

if 'master_df' not in st.session_state:
    st.session_state.master_df = load_master()

master_df = st.session_state.master_df

# Sidebar
with st.sidebar:
    st.title("⚡ Control Panel")
    st.markdown("---")
    
    st.subheader("1. Initialize Master")
    # CRITICAL: Let user pick date to match their CSV
    start_date = st.date_input("Start Date", value=date(2026, 1, 17))
    
    if st.button("Generate & Reset Master"):
        with st.spinner("Generating dataset using logic from sensor_database.ipynb..."):
            new_master = generate_full_dataset(start_date)
            new_master.to_csv(MASTER_CSV, index=False)
            st.session_state.master_df = new_master
            st.success(f"Generated {len(new_master)} records for {start_date}!")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.subheader("2. Update Master")
    uploaded_file = st.file_uploader("Upload CSV to Merge", type=['csv'])
    
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        
        # Simple column normalization just in case
        new_df.columns = new_df.columns.str.strip().str.title()
        rename_map = {'Time': 'Timestamp', 'Date': 'Timestamp', 'Sensor': 'Sensor_Name', 'Node': 'Sensor_Name'}
        new_df.rename(columns=rename_map, inplace=True)
        
        if 'Timestamp' in new_df.columns and 'Sensor_Name' in new_df.columns:
            if st.button("Merge Data"):
                updated_df, added, ignored = merge_new_file_into_master(master_df, new_df)
                
                updated_df.to_csv(MASTER_CSV, index=False)
                st.session_state.master_df = updated_df
                
                st.success("Update Successful!")
                st.markdown(f"""
                **Statistics (Logic from Update_Master.ipynb):**
                * Previous Master Size: `{len(master_df)}`
                * New File Size: `{len(new_df)}`
                * **Actually Added:** `{added}`
                * **Duplicates Ignored:** `{ignored}`
                * New Master Size: `{len(updated_df)}`
                """)
                time.sleep(2)
                st.rerun()
        else:
            st.error("Uploaded file must have 'Timestamp' and 'Sensor_Name' columns.")

    st.markdown("---")
    if st.button("⚠️ Delete Master File"):
        if os.path.exists(MASTER_CSV):
            os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value'])
        st.rerun()

# Main Dashboard
st.title("SensorEdge Pro (Notebook Logic)")

if not master_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(master_df):,}")
    col2.metric("Sensors", master_df['Sensor_Name'].nunique())
    
    # Calculate Time Range
    t_min = master_df['Timestamp'].min()
    t_max = master_df['Timestamp'].max()
    col3.metric("Time Span", f"{t_max - t_min}")

    st.divider()
    
    # Preview Data
    st.subheader("Data Preview")
    st.dataframe(master_df.head(100), use_container_width=True)
    
    st.subheader("Recent Data")
    st.dataframe(master_df.tail(100), use_container_width=True)
    
    st.download_button(
        label="Download Master CSV",
        data=master_df.to_csv(index=False).encode('utf-8'),
        file_name=MASTER_CSV,
        mime='text/csv'
    )

else:
    st.info("Master dataset is empty. Use the sidebar to Generate or Upload data.")
