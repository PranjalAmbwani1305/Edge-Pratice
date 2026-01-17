import streamlit as st
import pandas as pd
import os
from datetime import time

st.set_page_config(page_title="CSV Merge System", layout="centered")
st.title("Daily CSV Merge with Master (Ascending Date Order)")

MASTER_FILE = "master_dataset.csv"
MASTER_CUTOFF = time(12, 0, 0)

# ---------------------------------
# Load master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    master_df = pd.read_csv(MASTER_FILE, parse_dates=["timestamp"])
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value"]
    )

# Build existing key set
existing_keys = set(
    master_df["timestamp"].astype(str) + "_" + master_df["sensor"]
)

# ---------------------------------
# Upload CSV
# ---------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
        else:
            # Detect start time
            start_time = df["timestamp"].min().time()
            st.info(f"Detected CSV start time: {start_time}")

            # Filter relevant window (start → 12:00)
            df = df[
                (df["timestamp"].dt.time >= start_time) &
                (df["timestamp"].dt.time < MASTER_CUTOFF)
            ]

            st.write("Records after time filtering:", len(df))

            # Remove common records
            df_keys = df["timestamp"].astype(str) + "_" + df["sensor"]
            df = df[~df_keys.isin(existing_keys)]

            st.success(f"New records added: {len(df)}")

            # Merge
            master_df = pd.concat([master_df, df], ignore_index=True)

            # ✅ SORT BY DATE ASCENDING (IMPORTANT)
            master_df.sort_values(
                by="timestamp",
                ascending=True,
                inplace=True
            )

            # Save
            master_df.to_csv(MASTER_FILE, index=False)

            # Update key set
            existing_keys.update(
                df["timestamp"].astype(str) + "_" + df["sensor"]
            )

            st.subheader("Master Dataset (Ascending Date Order)")
            st.write("Total records:", len(master_df))
            st.dataframe(master_df.tail(15))

    except Exception as e:
        st.error(f"Error: {e}")

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
