import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
MASTER_CSV = 'master_dataset.csv'
V_REF = 5.0        # Sensor Reference Voltage
ADC_MAX = 1023     # 10-bit ADC Resolution

# --- Page Config ---
st.set_page_config(page_title="Edge Sensor Manager & Analytics", layout="wide", page_icon="📡")

# --- Helper Functions ---
def load_master():
    """Loads the master dataset if it exists."""
    if os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    return pd.DataFrame(columns=['Timestamp', 'Sensor_Name', 'Voltage_V', 'ADC_Value', 'Status'])

def highlight_rows(row):
    """Pandas Styler function for Excel export."""
    status = row.get('Status', '')
    if status == 'New':
        return ['background-color: #90EE90'] * len(row)  # Light Green
    elif status == 'Overlap':
        return ['background-color: #FF7F7F'] * len(row)  # Light Red
    elif status == 'Historical':
        return ['background-color: #FFFFFF'] * len(row)  # White
    return [''] * len(row)

def to_excel(df):
    """Converts dataframe to a color-coded Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.style.apply(highlight_rows, axis=1).to_excel(writer, index=False, sheet_name='SensorData')
    return output.getvalue()

def calculate_accuracy_metrics(df):
    """Calculates ADC consistency and ML Model Accuracy."""
    if df.empty or len(df) < 10:
        return 0.0, 0.0, "Not enough data"

    # 1. Hardware Accuracy (ADC Consistency)
    # Formula: Does the digital count match the analog voltage?
    # We allow a tiny margin of error (+/- 1 bit) for rounding differences.
    expected_adc = (df['Voltage_V'] / V_REF * ADC_MAX).astype(int)
    # Check if actual ADC is within 1 bit of expected
    matches = (abs(df['ADC_Value'] - expected_adc) <= 1)
    hardware_acc = matches.mean() * 100

    # 2. Model Accuracy (Sensor Identification)
    # Can we guess the Sensor Name purely from Voltage?
    try:
        X = df[['Voltage_V']]
        y = df['Sensor_Name']
        
        # We need at least 2 classes to classify
        if y.nunique() < 2:
            return hardware_acc, 0.0, "Need >1 Sensor Type"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        
        model_acc = accuracy_score(y_test, predictions) * 100
        return hardware_acc, model_acc, "Success"
    except Exception as e:
        return hardware_acc, 0.0, str(e)

# --- Main App Interface ---
st.title("📡 Edge Sensor Manager & Analytics")
st.markdown("### Dashboard")

# Load Data
master_df = load_master()

# Top Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(master_df))
if not master_df.empty:
    col2.metric("Start Time", master_df['Timestamp'].min().strftime('%H:%M:%S'))
    col3.metric("End Time", master_df['Timestamp'].max().strftime('%H:%M:%S'))
    col4.metric("Active Sensors", master_df['Sensor_Name'].nunique())
else:
    st.warning("No data found. Upload a CSV file to get started.")

st.divider()

# --- Layout: Two Columns (Upload vs Analytics) ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Edge Sensor CSV", type=['csv'])

    if uploaded_file is not None:
        st.info("Processing file...")
        
        # --- MERGE LOGIC ---
        new_df = pd.read_csv(uploaded_file)
        new_df['Timestamp'] = pd.to_datetime(new_df['Timestamp'], errors='coerce')
        new_df['Sensor_Name'] = new_df['Sensor_Name'].astype(str).str.strip()
        new_df['Status'] = 'New'  # Default to Green

        if not master_df.empty:
            # 1. Mark old master as Historical
            master_df['Status'] = 'Historical'
            
            # 2. Detect Overlaps
            master_keys = set(zip(master_df['Timestamp'], master_df['Sensor_Name']))
            new_keys = set(zip(new_df['Timestamp'], new_df['Sensor_Name']))
            overlap_keys = master_keys.intersection(new_keys)
            
            # Apply Red status to Master rows that overlap
            master_df['Status'] = master_df.apply(
                lambda x: 'Overlap' if (x['Timestamp'], x['Sensor_Name']) in overlap_keys else 'Historical', axis=1
            )
        
        # 3. Combine
        combined_df = pd.concat([master_df, new_df])
        combined_df = combined_df.sort_values(by='Timestamp')
        
        # 4. Deduplicate (Keep 'first' -> keeps the Red Overlap from Master, drops the Green Duplicate)
        final_df = combined_df.drop_duplicates(subset=['Timestamp', 'Sensor_Name'], keep='first').reset_index(drop=True)
        
        # 5. Save
        final_df.to_csv(MASTER_CSV, index=False)
        
        st.success(f"Merged successfully! Total rows: {len(final_df)}")
        st.experimental_rerun()

with right_col:
    st.subheader("🧠 Intelligence & Accuracy")
    if not master_df.empty:
        # Calculate Accuracy on the fly
        hw_acc, model_acc, status_msg = calculate_accuracy_metrics(master_df)
        
        ac1, ac2 = st.columns(2)
        ac1.metric(
            label="Hardware Accuracy (ADC)", 
            value=f"{hw_acc:.2f}%", 
            help="Checks if ADC_Value matches the Voltage_V formula."
        )
        ac2.metric(
            label="Model Prediction Accuracy", 
            value=f"{model_acc:.2f}%",
            help="Can AI predict the Sensor Type just from Voltage?"
        )
        
        if model_acc < 85:
            st.caption("⚠️ **Insight:** Low Model Accuracy suggests sensors (like Light/Temp) have overlapping voltage ranges.")
        else:
            st.caption("✅ **Insight:** Distinct voltage ranges allow for high identification accuracy.")

# --- Visualization Section ---
st.divider()
st.subheader("📈 Real-time Sensor Visualization")

if not master_df.empty:
    # Filter by sensor type
    sensors = master_df['Sensor_Name'].unique().tolist()
    selected_sensors = st.multiselect("Select Sensors to View", sensors, default=sensors)
    
    if selected_sensors:
        filtered_df = master_df[master_df['Sensor_Name'].isin(selected_sensors)]
        
        # Pivot for chart: Index=Time, Columns=Sensor, Values=Voltage
        chart_data = filtered_df.pivot_table(index='Timestamp', columns='Sensor_Name', values='Voltage_V', aggfunc='first')
        st.line_chart(chart_data)
    
    # --- Data Table & Export ---
    st.subheader("📋 Master Dataset View")
    
    # Quick Status Filter
    status_filter = st.radio("Show Rows:", ["All", "New (Green)", "Overlap (Red)", "Historical (White)"], horizontal=True)
    
    view_df = master_df.copy()
    if status_filter == "New (Green)":
        view_df = view_df[view_df['Status'] == 'New']
    elif status_filter == "Overlap (Red)":
        view_df = view_df[view_df['Status'] == 'Overlap']
    elif status_filter == "Historical (White)":
        view_df = view_df[view_df['Status'] == 'Historical']

    # Color mapping for UI
    def color_status(val):
        if val == 'New': return 'background-color: #90EE90; color: black'
        elif val == 'Overlap': return 'background-color: #FF7F7F; color: black'
        return ''

    st.dataframe(view_df.style.applymap(color_status, subset=['Status']), use_container_width=True)

    # Download Button
    st.download_button(
        label="📥 Download Colored Excel Report",
        data=to_excel(master_df),
        file_name='master_sensor_report.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# --- Reset Button (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Admin")
    if st.button("⚠️ Factory Reset Data"):
        if os.path.exists(MASTER_CSV):
            os.remove(MASTER_CSV)
        st.experimental_rerun()
