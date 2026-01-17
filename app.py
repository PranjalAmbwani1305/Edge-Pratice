import streamlit as st
import pandas as pd
import os
from datetime import time

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="Daily CSV Ingestion", layout="centered")
st.title("Daily CSV Ingestion (12–12 Master Dataset)")
st.write("Uploads replace overlapping data and keep the master dataset clean and ordered.")

MASTER_FILE = "master_dataset.csv"
MASTER_CUTOFF = time(12, 0, 0)

# ---------------------------------
# Load or initialize master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    master_df = pd.read_csv(MASTER_FILE, parse_dates=["timestamp"])
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value"]
    )

# ---------------------------------
# Upload CSV
# ---------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        # Validate schema
        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
        else:
            # ---------------------------------
            # Detect CSV start time
            # ---------------------------------
            start_time = df["timestamp"].min().time()
            st.info(f"Detected CSV start time: {start_time}")

            # ---------------------------------
            # Keep only start_time → 12:00
            # ---------------------------------
            df = df[
                (df["timestamp"].dt.time >= start_time) &
                (df["timestamp"].dt.time < MASTER_CUTOFF)
            ]

            st.write("Records after time filtering:", len(df))

            # ---------------------------------
            # REPLACE OVERLAP (LATEST WINS)
            # ---------------------------------
            df_keys = df["timestamp"].astype(str) + "_" + df["sensor"]

            # Remove overlapping rows from master
            master_df = master_df[
                ~(
                    master_df["timestamp"].astype(str) + "_" +
                    master_df["sensor"]
                ).isin(df_keys)
            ]

            # Append new data
            master_df = pd.concat([master_df, df], ignore_index=True)

            # ---------------------------------
            # Sort ASCENDING by timestamp
            # ---------------------------------
            master_df.sort_values(
                by="timestamp",
                ascending=True,
                inplace=True
            )

            # Save master dataset
            master_df.to_csv(MASTER_FILE, index=False)

            st.success(f"Records added / replaced: {len(df)}")

            # ---------------------------------
            # Summary
            # ---------------------------------
            st.subheader("Master Dataset Summary")
            st.write("Total records:", len(master_df))
            st.dataframe(master_df.tail(15))

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------------------------
# Download master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    with open(MASTER_FILE, "rb") as f:
        st.download_button(
            label="Download Master Dataset",
            data=f,
            file_name="master_dataset.csv",
            mime="text/csv"
        )
