import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

MASTER_FILE = "master_dataset.csv"

st.set_page_config(page_title="Master Dataset Merger", layout="centered")
st.title("Master Dataset (24-Hour) – Excel-Style Replace")

# -------------------------------------------------
# STEP 1: Generate random 24-hour master (one time)
# -------------------------------------------------
def generate_master():
    sensors = {
        "temperature": 2,            # seconds
        "light": 5,                  # seconds
        "moisture": 4 * 60 * 60      # 4 hours
    }

    start = datetime.strptime("2026-01-17 00:00:00", "%Y-%m-%d %H:%M:%S")
    end = start + timedelta(hours=24)

    rows = []

    for sensor, interval in sensors.items():
        t = start
        while t < end:
            value = round(np.random.uniform(10, 100), 2)
            rows.append([t, sensor, value])
            t += timedelta(seconds=interval)

    df = pd.DataFrame(rows, columns=["Timestamp", "Sensor", "Value"])
    df.to_csv(MASTER_FILE, index=False)
    return df


# -------------------------------------------------
# Load or create master
# -------------------------------------------------
if not os.path.exists(MASTER_FILE):
    master_df = generate_master()
else:
    master_df = pd.read_csv(MASTER_FILE)
    master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"], errors="coerce")


st.subheader("Current Master Dataset")
st.write("Total rows:", len(master_df))
st.dataframe(master_df.tail(10))

st.divider()

# -------------------------------------------------
# STEP 2: Upload CSV and merge (SAFE)
# -------------------------------------------------
uploaded = st.file_uploader(
    "Upload CSV (timestamp + sensor + value)",
    type="csv"
)

if uploaded:
    # Read WITHOUT parse_dates
    new_df = pd.read_csv(uploaded)

    # Normalize column names
    new_df.columns = [c.strip().lower() for c in new_df.columns]

    # Detect timestamp column
    if "timestamp" not in new_df.columns:
        st.error("CSV must contain a timestamp column")
        st.stop()

    if "sensor" not in new_df.columns or "value" not in new_df.columns:
        st.error("CSV must contain sensor and value columns")
        st.stop()

    # Convert timestamp safely
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], errors="coerce")

    # Rename to master schema
    new_df = new_df.rename(columns={
        "timestamp": "Timestamp",
        "sensor": "Sensor",
        "value": "Value"
    })

    # CONCAT (NEW AFTER OLD)
    combined = pd.concat([master_df, new_df], ignore_index=True)

    # 🔑 EXCEL-STYLE REPLACE
    combined = combined.drop_duplicates(
        subset=["Timestamp", "Sensor"],
        keep="last"
    )

    combined = combined.sort_values("Timestamp").reset_index(drop=True)

    combined.to_csv(MASTER_FILE, index=False)
    master_df = combined

    st.success("Merged successfully (duplicates replaced)")
    st.write("Total rows after merge:", len(master_df))
    st.dataframe(master_df.tail(20))

st.divider()

# -------------------------------------------------
# Download master
# -------------------------------------------------
st.download_button(
    "Download Master CSV",
    open(MASTER_FILE, "rb"),
    file_name="master_dataset.csv"
)
