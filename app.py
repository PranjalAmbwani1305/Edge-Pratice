import os
import pandas as pd
import streamlit as st
from datetime import time

# -------------------------------------------------
# App Configuration
# -------------------------------------------------
st.set_page_config(page_title="Master Dataset Manager", layout="centered")
st.title("Master Dataset Manager (Flask Logic, Stable Schema)")

UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MASTER_START = time(12, 0, 0)
MASTER_END = time(12, 0, 0)

# -------------------------------------------------
# Utility: normalize column names
# -------------------------------------------------
def normalize(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# -------------------------------------------------
# Utility: enforce 12–12 window
# -------------------------------------------------
def apply_12_12(df):
    t = df["timestamp"].dt.time
    return df[(t >= MASTER_START) | (t < MASTER_END)]

# -------------------------------------------------
# Summary (SAFE)
# -------------------------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None

    df = pd.read_csv(MASTER_CSV)
    df = normalize(df)

    # REQUIRED columns check
    if "timestamp" not in df.columns or "sensor" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = apply_12_12(df)

    return {
        "total_rows": len(df),
        "start_time": df["timestamp"].min(),
        "end_time": df["timestamp"].max(),
        "breakdown": df["sensor"].value_counts().to_dict(),
        "status_counts": (
            df["status"].value_counts().to_dict()
            if "status" in df.columns else {}
        )
    }

# -------------------------------------------------
# Load master safely
# -------------------------------------------------
if os.path.exists(MASTER_CSV):
    master_df = pd.read_csv(MASTER_CSV)
    master_df = normalize(master_df)
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value", "status"]
    )

master_df["timestamp"] = pd.to_datetime(master_df["timestamp"], errors="coerce")
master_df = apply_12_12(master_df)

# -------------------------------------------------
# Show summary
# -------------------------------------------------
summary = get_summary()

if summary:
    st.subheader("Master Dataset Summary (12–12)")
    st.write("Total Rows:", summary["total_rows"])
    st.write("Start Time:", summary["start_time"])
    st.write("End Time:", summary["end_time"])
    st.write("Sensor Breakdown:")
    st.json(summary["breakdown"])
    st.write("Status Breakdown:")
    st.json(summary["status_counts"])
else:
    st.info("No valid master dataset yet.")

st.divider()

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)
    new_df = normalize(new_df)

    required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
    if not required_cols.issubset(new_df.columns):
        st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
        st.stop()

    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], errors="coerce")
    new_df["sensor"] = new_df["sensor"].astype(str).str.strip()
    new_df["status"] = "New"

    new_df = apply_12_12(new_df)

    # Reset master status
    if not master_df.empty:
        master_df["status"] = "Historical"

        # Detect overlap
        master_keys = set(zip(master_df["timestamp"], master_df["sensor"]))
        new_keys = set(zip(new_df["timestamp"], new_df["sensor"]))
        overlap = master_keys.intersection(new_keys)

        master_df.loc[
            master_df.apply(
                lambda r: (r["timestamp"], r["sensor"]) in overlap, axis=1
            ),
            "status"
        ] = "Overlap"

    # Merge (KEEP MASTER ON DUPLICATE)
    combined = pd.concat([master_df, new_df], ignore_index=True)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(
        subset=["timestamp", "sensor"],
        keep="first"
    ).reset_index(drop=True)

    combined.to_csv(MASTER_CSV, index=False)
    master_df = combined

    st.success("CSV merged successfully (Flask logic preserved)")
    st.dataframe(master_df.tail(20))

st.divider()

# -------------------------------------------------
# Download
# -------------------------------------------------
if os.path.exists(MASTER_CSV):
    st.download_button(
        "Download Master CSV",
        open(MASTER_CSV, "rb"),
        file_name="master_dataset.csv"
    )

# -------------------------------------------------
# Reset
# -------------------------------------------------
if st.button("Reset Master Dataset"):
    if os.path.exists(MASTER_CSV):
        os.remove(MASTER_CSV)
    st.success("Master dataset reset. Reload page.")
