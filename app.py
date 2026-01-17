import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
MASTER_CSV = 'master_dataset.csv'
V_REF = 5.0        
ADC_MAX = 1023     

st.set_page_config(page_title="Sensor Manager & Accuracy", layout="wide")
st.title("📡 Sensor Manager & Analytics")

# --- 1. Helper Functions ---
def load_master():
    if os.path.exists(MASTER_CSV):
        try:
            df = pd.read_csv(MASTER_CSV)
            # Ensure Timestamp is datetime
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except:
            return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])

def calculate_accuracy(df):
    """Calculates Hardware (ADC) and AI Model Accuracy."""
    metrics = {"hw_acc": 0.0, "ai_acc": 0.0, "status": "No Data"}
    
    if df.empty or 'Voltage_V' not in df.columns: 
        return metrics

    # 1. Hardware Accuracy (ADC Logic Check)
    try:
        # Drop rows with missing values for calculation
        df_clean = df.dropna(subset=['Voltage_V', 'ADC_Value'])
        if not df_clean.empty:
            # Formula: ADC = (Voltage / 5.0) * 1023
            expected_adc = (df_clean['Voltage_V'] / V_REF * ADC_MAX).astype(int)
            # Check if actual ADC is within +/- 1 bit of expected
            matches = (abs(df_clean['ADC_Value'] - expected_adc) <= 1)
            metrics["hw_acc"] = matches.mean() * 100
    except:
        pass

    # 2. Model Accuracy (Can AI guess the sensor?)
    try:
        # We need at least 2 different sensors to train a classifier
        if df_clean['Sensor_Name'].nunique() >= 2:
            X = df_clean[['Voltage_V']]
            y = df_clean['Sensor_Name']
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Random Forest Model
            clf = RandomForestClassifier(n_estimators=20, random_state=42)
            clf.fit(X_train, y_train)
            
            # Predict
            predictions = clf.predict(X_test)
            metrics["ai_acc"] = accuracy_score(y_test, predictions) * 100
            metrics["status"] = "Active"
        else:
            metrics["status"] = "Need >1 Sensor"
    except Exception as e:
        metrics["status"] = "Model Error"
        
    return metrics

# --- 2. Load Data ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = load_master()

master_df = st.session_state.master_df
analytics = calculate_accuracy(master_df)

# --- 3. Sidebar: Upload & Merge ---
with st.sidebar:
    st.header("Actions")
    uploaded_file = st.file_uploader("Upload New CSV", type=['csv'])

    if uploaded_file:
        try:
            # Read and standardize
            new_df = pd.read_csv(uploaded_file)
            new_df.columns = new_df.columns.str.strip().str.title()
            rename_map = {
                'Time': 'Timestamp', 'Date': 'Timestamp', 
                'Sensor': 'Sensor_Name', 'Name': 'Sensor_Name', 
                'Voltage': 'Voltage_V', 'Volts': 'Voltage_V', 
                'Adc': 'ADC_Value'
            }
            new_df.rename(columns=rename_map, inplace=True)

            if 'Timestamp' in new_df.columns and 'Sensor_Name' in new_df.columns:
                # --- LOGIC START ---
                new_df['Status'] = 'New' # New data is Green
                
                if not master_df.empty:
                    master_df['Status'] = 'Historical' # Old data is White
                    
                    # Fast Vectorized Overlap Check
                    master_idx = pd.MultiIndex.from_frame(master_df[['Timestamp', 'Sensor_Name']])
                    new_idx = pd.MultiIndex.from_frame(new_df[['Timestamp', 'Sensor_Name']])
                    
                    # Mark Overlaps as Red
                    overlap_mask = master_idx.isin(new_idx)
                    master_df.loc[overlap_mask, 'Status'] = 'Overlap'

                # Merge: Master (Red/White) on top of New (Green)
                combined_df = pd.concat([master_df, new_df])
                combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
                combined_df = combined_df.sort_values(by='Timestamp')
                
                # Keep first (Master wins)
                final_df = combined_df.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first')
                # --- LOGIC END ---

                final_df.to_csv(MASTER_CSV, index=False)
                st.session_state.master_df = final_df
                st.success("Merged Successfully!")
                st.rerun()
            else:
                st.error("Missing columns: Timestamp, Sensor_Name")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    if st.button("Reset All Data"):
        if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
        st.session_state.master_df = pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])
        st.rerun()

# --- 4. Main Dashboard ---

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(master_df))
col2.metric("System Status", analytics['status'])
col3.metric("Hardware Accuracy", f"{analytics['hw_acc']:.1f}%", help="Does ADC match Voltage?")
col4.metric("AI Model Accuracy", f"{analytics['ai_acc']:.1f}%", help="Can AI identify sensor by voltage?")

st.divider()

# Charts
if not master_df.empty and 'Voltage_V' in master_df.columns:
    st.subheader("Sensor Visualization")
    sensors = master_df['Sensor_Name'].unique().tolist()
    sel = st.multiselect("Select Sensors", sensors, default=sensors)
    if sel:
        chart_data = master_df[master_df['Sensor_Name'].isin(sel)].pivot_table(index='Timestamp', columns='Sensor_Name', values='Voltage_V', aggfunc='first')
        st.line_chart(chart_data)

# Data Table
st.subheader("Master Dataset")

def color_status(val):
    if val == 'New': return 'background-color: #90EE90; color: black'   # Green
    elif val == 'Overlap': return 'background-color: #FF7F7F; color: black' # Red
    return '' # White

if not master_df.empty:
    # Use generic styling to avoid errors
    try:
        st.dataframe(master_df.style.map(color_status, subset=['Status']), use_container_width=True)
    except:
        st.dataframe(master_df.style.applymap(color_status, subset=['Status']), use_container_width=True)
        
    st.download_button(
        "Download Master CSV", 
        master_df.to_csv(index=False).encode('utf-8'), 
        "master_dataset.csv", 
        "text/csv"
    )
