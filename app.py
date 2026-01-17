import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
MASTER_CSV = 'master_dataset.csv'
V_REF = 5.0        
ADC_MAX = 1023     

# --- Page Config ---
st.set_page_config(page_title="Sensor Manager", layout="wide")

# --- Helper Functions ---
def load_master():
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except:
            return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])

def standardize_columns(df):
    """Renames columns to standard format."""
    df.columns = df.columns.str.strip().str.lower()
    col_map = {
        'time': 'Timestamp', 'date': 'Timestamp', 'datetime': 'Timestamp', 't': 'Timestamp',
        'sensor': 'Sensor_Name', 'sensor_name': 'Sensor_Name', 'name': 'Sensor_Name', 
        'node': 'Sensor_Name', 'id': 'Sensor_Name',
        'voltage': 'Voltage_V', 'volts': 'Voltage_V', 'v': 'Voltage_V',
        'adc': 'ADC_Value', 'adc_value': 'ADC_Value'
    }
    df.rename(columns=col_map, inplace=True)
    full_map = {'timestamp': 'Timestamp', 'sensor_name': 'Sensor_Name', 'voltage_v': 'Voltage_V', 'adc_value': 'ADC_Value'}
    df.rename(columns=full_map, inplace=True)
    return df

def generate_demo_data():
    """Generates 24-hour baseline data."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    duration_sec = 24 * 3600
    
    def make_sensor(name, interval, v_min, v_max):
        times = [start_time + timedelta(seconds=i) for i in range(0, duration_sec, interval)]
        volts = np.random.uniform(v_min, v_max, len(times))
        adcs = (volts / V_REF * ADC_MAX).astype(int)
        return pd.DataFrame({
            'Timestamp': times, 'Sensor_Name': name, 
            'Voltage_V': np.round(volts, 4), 'ADC_Value': adcs, 'Status': 'Historical'
        })
    
    df1 = make_sensor("Temperature", 2, 2.0, 4.0)
    df2 = make_sensor("Light", 5, 0.0, 5.0)
    df3 = make_sensor("Moisture", 14400, 1.2, 3.0)
    
    master = pd.concat([df1, df2, df3]).sort_values('Timestamp').reset_index(drop=True)
    master.to_csv(MASTER_CSV, index=False)
    return master

def calculate_analytics(df):
    metrics = {"hw_acc": 0.0, "ai_acc": 0.0, "status": "No Data"}
    if df.empty or 'Voltage_V' not in df.columns: return metrics

    try:
        df_clean = df.dropna(subset=['Voltage_V', 'ADC_Value'])
        if not df_clean.empty:
            expected = (df_clean['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            metrics["hw_acc"] = (abs(df_clean['ADC_Value'] - expected) <= 1).mean() * 100
    except: pass

    try:
        if df_clean['Sensor_Name'].nunique() >= 2:
            X = df_clean[['Voltage_V']]
            y = df_clean['Sensor_Name']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=20, random_state=42)
            clf.fit(X_tr, y_tr)
            metrics["ai_acc"] = accuracy_score(y_te, clf.predict(X_te)) * 100
            metrics["status"] = "Healthy"
        else:
            metrics["status"] = "Calibrating..."
    except:
        metrics["status"] = "Error"
    return metrics

def to_excel(df):
    output = io.BytesIO()
    # Simple function without fancy styling
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='SensorData')
    return output.getvalue()

# --- MAIN APP ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_master()

master_df = st.session_state.master_df
analytics = calculate_analytics(master_df)

# --- Sidebar ---
with st.sidebar:
    st.header("Actions")
    
    if st.button("Initialize 24h Demo Data"):
        with st.spinner("Generating..."):
            st.session_state.master_df = generate_demo_data()
            st.rerun()

    st.divider()
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file)
            new_df = standardize_columns(new_df)
            
            if 'Timestamp' not in new_df.columns or 'Sensor_Name' not in new_df.columns:
                st.error("Missing Timestamp or Sensor_Name columns.")
            else:
                new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'], errors='coerce')
                new_df = new_df.dropna(subset=['Timestamp'])
                new_df['Sensor_Name'] = new_df['Sensor_Name'].astype(str).str.strip()
                if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
                
                new_df['Status'] = 'New'
                if not master_df.empty:
                    master_df['Status'] = 'Historical'
                    # Simple overlap logic
                    keys_m = set(zip(master_df['Timestamp'], master_df['Sensor_Name']))
                    keys_n = set(zip(new_df['Timestamp'], new_df['Sensor_Name']))
                    overlap = keys_m.intersection(keys_n)
                    master_df['Status'] = master_df.apply(lambda x: 'Overlap' if (x['Timestamp'], x['Sensor_Name']) in overlap else 'Historical', axis=1)

                final = pd.concat([master_df, new_df]).sort_values('Timestamp').drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
                final.to_csv(MASTER_CSV, index=False)
                st.session_state.master_df = final
                st.success("File Merged.")
                time.sleep(1)
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    if st.button("Reset Data"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
        st.rerun()

# --- Main Layout ---
st.title("Sensor Manager")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(master_df))
col2.metric("System Status", analytics['status'])
col3.metric("Hardware Accuracy", f"{analytics['hw_acc']:.1f}%")
col4.metric("AI Accuracy", f"{analytics['ai_acc']:.1f}%")

st.divider()

# Charts & Data
if not master_df.empty:
    st.subheader("Sensor Visualization")
    if 'Voltage_V' in master_df.columns:
        sensors = master_df['Sensor_Name'].unique().tolist()
        selected = st.multiselect("Select Sensors", sensors, default=sensors)
        if selected:
            chart_data = master_df[master_df['Sensor_Name'].isin(selected)].pivot_table(index='Timestamp', columns='Sensor_Name', values='Voltage_V', aggfunc='first')
            st.line_chart(chart_data)

    st.subheader("Data Table")
    # Simple table
    st.dataframe(master_df, use_container_width=True)
    
    st.download_button("Download CSV", master_df.to_csv(index=False), "sensor_data.csv", "text/csv")
else:
    st.info("No data loaded. Use the sidebar to generate demo data or upload a file.")
