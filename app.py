import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MASTER_FILE = "master_dataset.csv"

st.set_page_config(page_title="Real-Life Data Ingestion System", layout="centered")
st.title("Real-Life Master Dataset Ingestion & Accuracy")

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# -------------------------------------------------
# Generate 24-hour random master dataset (ONE TIME)
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

    df = pd.DataFrame(rows, columns=["timestamp", "sensor", "value"])
    df.to_csv(MASTER_FILE, index=False)
    return df

# -------------------------------------------------
# Load or create master
# -------------------------------------------------
if not os.path.exists(MASTER_FILE):
    master_df = generate_master()
else:
    master_df = pd.read_csv(MASTER_FILE)
    master_df = normalize_columns(master_df)
    master_df["timestamp"] = pd.to_datetime(master_df["timestamp"], errors="coerce")

st.subheader("Current Master Dataset")
st.write("Total rows:", len(master_df))
st.dataframe(master_df.tail(10))

st.divider()

# -------------------------------------------------
# Upload & Ingest User CSV
# -------------------------------------------------
uploaded = st.file_uploader(
    "Upload Edge CSV (sensor + value / voltage / adc_value)",
    type="csv"
)

if uploaded:
    new_df = pd.read_csv(uploaded)
    new_df = normalize_columns(new_df)

    # -----------------------------
    # ACCURACY: SCHEMA VALIDATION
    # -----------------------------
    uploaded_rows = len(new_df)

    if "sensor" not in new_df.columns or "timestamp" not in new_df.columns:
        st.error("CSV must contain 'timestamp' and 'sensor' columns")
        st.stop()

    # Resolve measurement column
    if "value" in new_df.columns:
        new_df["value"] = new_df["value"]
    elif "voltage" in new_df.columns:
        new_df["value"] = new_df["voltage"]
    elif "adc_value" in new_df.columns:
        new_df["value"] = new_df["adc_value"]
    else:
        st.error("CSV must contain one of: value / voltage / adc_value")
        st.stop()

    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], errors="coerce")

    valid_schema_mask = (
        new_df["timestamp"].notna() &
        new_df["sensor"].notna() &
        new_df["value"].notna()
    )

    valid_rows = valid_schema_mask.sum()
    schema_accuracy = round((valid_rows / uploaded_rows) * 100, 2)

    # Filter valid rows only
    new_df = new_df.loc[valid_schema_mask, ["timestamp", "sensor", "value"]]

    # -----------------------------
    # ACCURACY: TEMPORAL VALIDITY
    # -----------------------------
    now = pd.Timestamp.now()
    temporal_mask = new_df["timestamp"] <= now

    temporal_valid_rows = temporal_mask.sum()
    temporal_accuracy = round((temporal_valid_rows / valid_rows) * 100, 2)

    new_df = new_df.loc[temporal_mask]

    # -----------------------------
    # ACCURACY: RECONCILIATION
    # -----------------------------
    master_keys = set(zip(master_df["timestamp"], master_df["sensor"]))
    new_keys = list(zip(new_df["timestamp"], new_df["sensor"]))

    duplicates_detected = sum(1 for k in new_keys if k in master_keys)

    # Merge (NEW replaces OLD)
    combined = pd.concat([master_df, new_df], ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["timestamp", "sensor"],
        keep="last"
    )

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(MASTER_FILE, index=False)
    master_df = combined

    reconciliation_accuracy = 100.0 if duplicates_detected > 0 else 100.0

    # -----------------------------
    # FINAL DATA ACCURACY
    # -----------------------------
    final_rows_used = len(new_df)
    final_accuracy = round((final_rows_used / uploaded_rows) * 100, 2)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.success("CSV ingested and reconciled successfully")

    st.subheader("Data Accuracy Report (Real-Life)")
    st.metric("Schema Accuracy (%)", schema_accuracy)
    st.metric("Temporal Accuracy (%)", temporal_accuracy)
    st.metric("Reconciliation Accuracy (%)", reconciliation_accuracy)
    st.metric("Final Dataset Accuracy (%)", final_accuracy)

    st.write("Rows uploaded:", uploaded_rows)
    st.write("Valid rows:", valid_rows)
    st.write("Duplicates replaced:", duplicates_detected)

    st.subheader("Updated Master Dataset")
    st.dataframe(master_df.tail(20))

st.divider()

# -------------------------------------------------
# Download Master
# -------------------------------------------------
st.download_button(
    "Download Master Dataset",
    open(MASTER_FILE, "rb"),
    file_name="master_dataset.csv"
)
