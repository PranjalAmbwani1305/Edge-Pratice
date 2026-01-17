import streamlit as st
import pandas as pd
import os

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="Generic CSV Ingestion", layout="centered")
st.title("Generic Time-Series CSV Ingestion")

MASTER_FILE = "master_dataset.csv"

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
uploaded_file = st.file_uploader("Upload CSV (any timestamp range)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        # Validate schema
        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
        else:
            st.info(
                f"Uploaded range: {df['timestamp'].min()} → {df['timestamp'].max()}"
            )

            # ---------------------------------
            # Build overlap keys
            # ---------------------------------
            upload_keys = df["timestamp"].astype(str) + "_" + df["sensor"]
            master_keys = master_df["timestamp"].astype(str) + "_" + master_df["sensor"]

            # ---------------------------------
            # REMOVE overlapping records (latest wins)
            # ---------------------------------
            before_count = len(master_df)

            master_df = master_df[
                ~master_keys.isin(upload_keys)
            ]

            removed = before_count - len(master_df)

            # ---------------------------------
            # Append new data
            # ---------------------------------
            master_df = pd.concat([master_df, df], ignore_index=True)

            # ---------------------------------
            # Sort ascending by timestamp
            # ---------------------------------
            master_df.sort_values(
                by="timestamp",
                ascending=True,
                inplace=True
            )

            # Save master
            master_df.to_csv(MASTER_FILE, index=False)

            # ---------------------------------
            # Output summary
            # ---------------------------------
            st.success("CSV merged successfully")
            st.write(f"Overlapping records replaced: {removed}")
            st.write(f"New records added: {len(df)}")
            st.write(f"Total records in master: {len(master_df)}")

            st.subheader("Master Dataset (Latest Records)")
            st.dataframe(master_df.tail(20))

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------------------------
# Download master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    with open(MASTER_FILE, "rb") as f:
        st.download_button(
            "Download Master Dataset",
            data=f,
            file_name="master_dataset.csv",
            mime="text/csv"
        )
