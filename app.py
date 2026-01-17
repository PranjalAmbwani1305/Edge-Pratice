import os
import pandas as pd
import streamlit as st
from datetime import time

UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"
MASTER_EXCEL = "master_dataset_colored.xlsx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Time window (as requested)
START_TIME = time(3, 0, 0)   # 03:00
END_TIME   = time(12, 0, 0)  # 12:00

st.set_page_config(page_title="Master Dataset (Streamlit)", layout="centered")
st.title("Master Dataset Manager (Flask Logic + Time Window)")

# -------------------------------------------------
# 🔧 Column mapping (CRITICAL FIX)
# -------------------------------------------------
def map_to_flask_columns(df):
    rename_map = {
        "timestamp": "Timestamp",
        "sensor": "Sensor_Name",
        "status": "Status"
    }
    df.columns = [rename_map.get(c.lower(), c) for c in df.columns]
    return df

# -------------------------------------------------
# Summary (unchanged logic)
# -------------------------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None
    try:
        df = pd.read_csv(MASTER_CSV)
        df = map_to_flask_columns(df)

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        return {
            "total_rows": len(df),
            "start_time": df["Timestamp"].min(),
            "end_time": df["Timestamp"].max(),
            "breakdown": df["Sensor_Name"].value_counts().to_dict(),
            "status_counts": (
                df["Status"].value_counts().to_dict()
                if "Status" in df.columns else {}
            )
        }
    except Exception as e:
        st.error(f"Error reading summary: {e}")
        return None

# -------------------------------------------------
# Show summary
# -------------------------------------------------
summary = get_summary()
if summary:
    st.subheader("Master Dataset Summary")
    st.write("Total Rows:", summary["total_rows"])
    st.write("Start Time:", summary["start_time"])
    st.write("End Time:", summary["end_time"])
    st.json(summary["breakdown"])
    st.json(summary["status_counts"])
else:
    st.info("No master dataset yet.")

st.divider()

# -------------------------------------------------
# Upload CSV (FLASK LOGIC + TIME WINDOW)
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. LOAD NEW DATA
    new_df = pd.read_csv(filepath)
    new_df = map_to_flask_columns(new_df)

    # Safety check
    if "Timestamp" not in new_df.columns or "Sensor_Name" not in new_df.columns:
        st.error("CSV must contain timestamp and sensor columns")
        st.stop()

    new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], errors="coerce")
    new_df["Sensor_Name"] = new_df["Sensor_Name"].astype(str).str.strip()

    # 🔑 TIME WINDOW (03:00 → 12:00)
    new_df = new_df[
        (new_df["Timestamp"].dt.time >= START_TIME) &
        (new_df["Timestamp"].dt.time < END_TIME)
    ]

    new_df["Status"] = "New"

    # 2. LOAD MASTER
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df = map_to_flask_columns(master_df)

        master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"], errors="coerce")
        master_df["Sensor_Name"] = master_df["Sensor_Name"].astype(str).str.strip()
        master_df["Status"] = "Historical"
    else:
        master_df = pd.DataFrame()

    # 3. OVERLAP DETECTION (UNCHANGED)
    if not master_df.empty and not new_df.empty:
        master_keys = set(zip(master_df["Timestamp"], master_df["Sensor_Name"]))
        new_keys = set(zip(new_df["Timestamp"], new_df["Sensor_Name"]))
        overlap_keys = master_keys.intersection(new_keys)

        def mark_overlap(row):
            if (row["Timestamp"], row["Sensor_Name"]) in overlap_keys:
                return "Overlap"
            return "Historical"

        master_df["Status"] = master_df.apply(mark_overlap, axis=1)

    # 4. MERGE
    combined_df = pd.concat([master_df, new_df])

    # 5. SORT & DEDUP (MASTER WINS)
    combined_df = combined_df.sort_values("Timestamp")
    combined_df = combined_df.drop_duplicates(
        subset=["Timestamp", "Sensor_Name"],
        keep="first"
    ).reset_index(drop=True)

    # 6. SAVE
    combined_df.to_csv(MASTER_CSV, index=False)

    os.remove(filepath)

    st.success("CSV merged successfully")
    st.write("Total rows in master:", len(combined_df))
    st.dataframe(combined_df.tail(20))

st.divider()

# -------------------------------------------------
# Downloads & Reset
# -------------------------------------------------
if os.path.exists(MASTER_CSV):
    st.download_button(
        "Download Master CSV",
        open(MASTER_CSV, "rb"),
        file_name="master_dataset.csv"
    )

if st.button("Reset Master Dataset"):
    if os.path.exists(MASTER_CSV):
        os.remove(MASTER_CSV)
    st.success("Master dataset reset. Reload page.")
