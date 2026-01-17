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

# --- Page Config ---
st.set_page_config(
    page_title="SensorEdge Pro | Enterprise Dashboard", 
    layout="wide", 
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; border: 1px solid #e0e0e0;
        padding: 15px; border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

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
    """
    Smartly renames columns. 
    Includes 'node', 'id', 't' to catch your specific file format.
    """
    df.columns = df.columns.str.strip().str.lower() 
    
    col_map = {
        # Time variations
        'time': 'Timestamp', 'date': 'Timestamp', 'datetime': 'Timestamp', 't': 'Timestamp', 'epoch': 'Timestamp',
        # Sensor Name variations
        'sensor': 'Sensor_Name', 'sensor_name': 'Sensor_Name', 'name': 'Sensor_Name', 
        'node': 'Sensor_Name', 'node_id': 'Sensor_Name', 'id': 'Sensor_Name', 'source': 'Sensor_Name',
        # Voltage variations
        'voltage': 'Voltage_V', 'volts': 'Voltage_V', 'v': 'Voltage_V', 'val': 'Voltage_V',
        # ADC variations
        'adc': 'ADC_Value', 'adc_value': 'ADC_Value'
    }
    
    df.rename(columns=col_map, inplace=True)
    
    # Capitalize standard columns for display
    full_map = {
        'timestamp': 'Timestamp', 
        'sensor_name': 'Sensor_Name', 
        'voltage_v': 'Voltage_V', 
        'adc_value': 'ADC_Value'
    }
    df.rename(columns=full_map, inplace=True)
    return df

def calculate_analytics(df):
    metrics = {"hw_acc": 0.0, "ai_acc": 0.0, "status": "No Data", "color": "off"}
    if df.empty or 'Voltage_V' not in df.columns: return metrics

    try:
        df_clean = df.dropna(subset=['Voltage_V', 'ADC_Value'])
        if not df_clean.empty:
            expected_adc = (df_clean['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            matches = (abs(df_clean['ADC_Value'] - expected_adc) <= 1)
            metrics["hw_acc"] = matches.mean() * 100
    except: pass

    try:
        if df_clean['Sensor_Name'].nunique() >= 2:
            X = df_clean[['Voltage_V']]
            y = df_clean['Sensor_Name']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(n_estimators=20, random_state=42)
            clf.fit(X_train, y_train)
            metrics["ai_acc"] = accuracy_score(y_test, clf.predict(X_test)) * 100
            metrics["status"] = "System Healthy"
        else:
            metrics["status"] = "Calibrating..."
    except:
        metrics["status"] = "Model Error"
    return metrics

def to_excel(df):
    output = io.BytesIO()
    def highlight_rows(row):
        status = row.get('Status', '')
        if status == 'New': return ['background-color: #d4edda'] * len(row)
        elif status == 'Overlap': return ['background-color: #f8d7da'] * len(row)
        return [''] * len(row)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.style.apply(highlight_rows, axis=1).to_excel(writer, index=False, sheet_name='SensorData')
    return output.getvalue()

# --- MAIN APP ---
master_df = load_master()
analytics = calculate_analytics(master_df)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2906/2906274.png", width=50)
    st.title("Control Panel")
    
    st.subheader("📤 Ingest Data")
    uploaded_file = st.file_uploader("Drop CSV Here", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file)
            
            # 1. Standardize Names
            new_df = standardize_columns(new_df)
            
            # 2. Validation
            required = ['Timestamp', 'Sensor_Name']
            missing = [c for c in required if c not in new_df.columns]
            
            if missing:
                st.error("❌ File Format Error")
                st.write("**Missing Columns:**", missing)
                st.write("**Found Columns:**", list(new_df.columns))
            else:
                with st.spinner("Merging..."):
                    new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'], errors='coerce')
                    new_df = new_df.dropna(subset=['Timestamp'])
                    new_df['Sensor_Name'] = new_df['Sensor_Name'].astype(str).str.strip()
                    if 'Voltage_V' not in new_df.columns: new_df['Voltage_V'] = 0.0
                    
                    new_df['Status'] = 'New'
                    if not master_df.empty:
                        master_df['Status'] = 'Historical'
                        master_keys = set(zip(master_df['Timestamp'], master_df['Sensor_Name']))
                        new_keys = set(zip(new_df['Timestamp'], new_df['Sensor_Name']))
                        overlap = master_keys.intersection(new_keys)
                        master_df['Status'] = master_df.apply(
                            lambda x: 'Overlap' if (x['Timestamp'], x['Sensor_Name']) in overlap else 'Historical', axis=1
                        )

                    combined = pd.concat([master_df, new_df]).sort_values('Timestamp')
                    final = combined.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
                    final.to_csv(MASTER_CSV, index=False)
                    st.success("Sync Complete!")
                    time.sleep(1)
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("Factory Reset Database", type="secondary"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.rerun()

# --- DASHBOARD LAYOUT ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("SensorEdge Analytics")

# --- FIX APPLIED HERE: Clean IF/ELSE Block ---
with col_h2:
    if analytics['status'] == "System Healthy":
        st.success(f"● {analytics['status']}")
    else:
        st.warning(f"● {analytics['status']}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Records", f"{len(master_df):,}")
kpi3.metric("Hardware Fidelity", f"{analytics['hw_acc']:.1f}%")
kpi4.metric("AI Confidence", f"{analytics['ai_acc']:.1f}%")

st.markdown("---")
tab_charts, tab_data = st.tabs(["📊 Analytics", "📋 Data Inspector"])

with tab_charts:
    if not master_df.empty and 'Voltage_V' in master_df.columns:
        sensors = master_df['Sensor_Name'].unique().tolist()
        sel = st.multiselect("Select Sensors", sensors, default=sensors)
        if sel:
            chart_data = master_df[master_df['Sensor_Name'].isin(sel)].pivot_table(index='Timestamp', columns='Sensor_Name', values='Voltage_V', aggfunc='first')
            st.line_chart(chart_data, height=350)
    else:
        st.info("No data available for visualization.")

with tab_data:
    if not master_df.empty:
        status_filter = st.radio("Status:", ["All", "New", "Overlap", "Historical"], horizontal=True)
        view_df = master_df.copy()
        if status_filter != "All": 
            view_df = view_df[view_df['Status'].str.contains(status_filter.split()[0])]
        
        st.dataframe(
            view_df,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm:ss"),
                "Voltage_V": st.column_config.ProgressColumn("Voltage", min_value=0, max_value=5, format="%.2f V"),
            },
            use_container_width=True, hide_index=True
        )
        st.download_button("📥 Export Excel", data=to_excel(master_df), file_name="sensor_report.xlsx")
