import os
import pandas as pd
import streamlit as st
from datetime import time

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(page_title="Master Dataset Manager", layout="centered")
st.title("Master Dataset Manager (12–12 Canonical Window)")

UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MASTER_START = time(12, 0, 0)
MASTER_END   = time(12, 0, 0)

# ---------------------------------
# Utility: Normalize columns
# ---------------------------------
def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ---------------------------------
# Utility: Enforce 12–12 window
# ---------------------------------
def apply_12_12_window(df):
    """
    Keeps only timestamps belonging to 12:00 → next day 12:00
    """
    t = df["timestamp"].dt.time
    return df[(t >= MASTER_START) | (t < MASTER_END)]

# ---------------------------------
# Utility: Load Summary (12–12 aware)
# ---------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None

    df = pd.read_csv(MASTER_CSV)
    df = normalize_columns(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ENFORCE CANONICAL WINDOW BEFORE SUMMARY
    df = apply_12_12_window(df)

    return {
        "total_rows": len(df),
        "start_time": df["timestamp"].min(),
        "end_time": df["timestamp"].max(),
        "breakdown": df["sensor"].value_counts().to_dict()
    }

# ---------------------------------
# Load Master Dataset
# ---------------------------------
if os.path.exists(MASTER_CSV):
    master_df = pd.read_csv(MASTER_CSV)
    master_df = normalize_columns(master_df)
    master_df["timestamp"] = pd.to_datetime(master_df["timestamp"], errors="coerce")

    # ENFORCE 12–12 WINDOW ON LOAD
    master_df = apply_12_12_window(master_df)
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value"]
    )

# ---------------------------------
# Show Summary
# ---------------------------------
summary = get_summary()

if summary:
    st.subheader("Master Dataset Summary (12–12)")
    st.write("Total Rows:", summary["total_rows"])
    st.write("Start Time:", summary["start_time"])
    st.write("End Time:", summary["end_time"])
    st.json(summary["breakdown"])
else:
    st.info("No master dataset yet.")

st.divider()

# ---------------------------------
# Upload CSV
# ---------------------------------
uploaded_file = st.file_uploader("Upload CSV (any timestamp)", type=["csv"])

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    new_df = normalize_columns(new_df)

    required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
    if not required_cols.issubset(new_df.columns):
        st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
        st.stop()

    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], errors="coerce")
    new_df["sensor"] = new_df["sensor"].astype(str).str.strip()

    # ENFORCE 12–12 WINDOW ON UPLOAD
    new_df = apply_12_12_window(new_df)

    before = len(master_df)

    # ---------------------------------
    # CONCAT + REMOVE DUPLICATES
    # ---------------------------------
    combined_df = pd.concat([master_df, new_df], ignore_index=True)

    combined_df = combined_df.drop_duplicates(
        subset=["timestamp", "sensor"],
        keep="first"
    )

    # Sort ascending
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    # Save
    combined_df.to_csv(MASTER_CSV, index=False)
    master_df = combined_df

    st.success("CSV merged into 12–12 master dataset")
    st.write("New rows added:", len(master_df) - before)
    st.write("Total rows:", len(master_df))

    st.dataframe(master_df.tail(20))

# ---------------------------------
# Download
# ---------------------------------
if os.path.exists(MASTER_CSV):
    with open(MASTER_CSV, "rb") as f:
        st.download_button(
            "Download Master CSV",
            f,
            file_name="master_dataset.csv",
            mime="text/csv"
        )
