import streamlit as st
import pandas as pd
import os
from datetime import time

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title="Daily CSV Ingestion System",
    layout="centered"
)

st.title("Daily Sensor CSV Ingestion")
st.write("Upload any daily CSV. Data will be aligned to the 12:00 → 12:00 master dataset.")

MASTER_FILE = "master_dataset.csv"
MASTER_CUTOFF = time(12, 0, 0)

# ---------------------------------
# Load existing master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    master_df = pd.read_csv(MASTER_FILE, parse_dates=["timestamp"])
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value"]
    )

# Build lookup for overlap detection (REAL-LIFE LOGIC)
existing_keys = set(
    master_df["timestamp"].astype(str) + "_" + master_df["sensor"]
)

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        # Validate schema
        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(df.columns):
            st.error("Invalid CSV format. Required columns: timestamp, sensor, voltage, adc_value")
        else:
            # ---------------------------------
            # Auto-detect start time
            # ---------------------------------
            start_time = df["timestamp"].min().time()
            st.info(f"Detected CSV start time: {start_time}")

            # ---------------------------------
            # Keep only non-overlapping time window
            # (start_time → 12:00)
            # ---------------------------------
            df = df[
                (df["timestamp"].dt.time >= start_time) &
                (df["timestamp"].dt.time < MASTER_CUTOFF)
            ]

            st.write("Records after time filtering:", len(df))

            # ---------------------------------
            # Remove overlap BEFORE merge
            # ---------------------------------
            overlap_keys = (
                df["timestamp"].astype(str) + "_" + df["sensor"]
            )

            df = df[~overlap_keys.isin(existing_keys)]

            st.success(f"New records added: {len(df)}")

            # ---------------------------------
            # Append and save master (NO helper cols)
            # ---------------------------------
            master_df = pd.concat([master_df, df], ignore_index=True)
            master_df.sort_values("timestamp", inplace=True)
            master_df.to_csv(MASTER_FILE, index=False)

            # Update lookup
            existing_keys.update(
                df["timestamp"].astype(str) + "_" + df["sensor"]
            )

            # ---------------------------------
            # Final Summary
            # ---------------------------------
            st.subheader("Master Dataset Summary")
            st.write("Total records:", len(master_df))
            st.dataframe(master_df.tail(15))

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------------------------
# Download Master Dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    with open(MASTER_FILE, "rb") as f:
        st.download_button(
            label="Download Master Dataset",
            data=f,
            file_name="master_dataset.csv",
            mime="text/csv"
        )
