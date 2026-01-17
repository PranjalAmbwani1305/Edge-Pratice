import os
import pandas as pd
import streamlit as st
from datetime import time

# -------------------------------------------------
# Config
# -------------------------------------------------
st.set_page_config(page_title="Master Dataset (12–12)", layout="centered")
st.title("Master Dataset Manager (Flask Logic Preserved)")

UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"
MASTER_EXCEL = "master_dataset_colored.xlsx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MASTER_START = time(12, 0, 0)
MASTER_END = time(12, 0, 0)

# -------------------------------------------------
# Utility: enforce 12–12 window
# -------------------------------------------------
def apply_12_12(df):
    t = df["Timestamp"].dt.time
    return df[(t >= MASTER_START) | (t < MASTER_END)]

# -------------------------------------------------
# Summary (same logic as Flask, 12–12 aware)
# -------------------------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None

    df = pd.read_csv(MASTER_CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = apply_12_12(df)

    status_counts = (
        df["Status"].value_counts().to_dict()
        if "Status" in df.columns else {}
    )

    return {
        "total_rows": len(df),
        "start_time": df["Timestamp"].min(),
        "end_time": df["Timestamp"].max(),
        "breakdown": df["Sensor_Name"].value_counts().to_dict(),
        "status_counts": status_counts
    }

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
    st.info("No master dataset yet.")

st.divider()

# -------------------------------------------------
# Upload CSV (same as Flask)
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. NEW DATA (Green)
    new_df = pd.read_csv(filepath)
    new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], errors="coerce")
    new_df["Sensor_Name"] = new_df["Sensor_Name"].astype(str).str.strip()
    new_df["Status"] = "New"

    # 2. OLD MASTER (White / Red)
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"], errors="coerce")
        master_df["Sensor_Name"] = master_df["Sensor_Name"].astype(str).str.strip()
        master_df["Status"] = "Historical"
    else:
        master_df = pd.DataFrame()

    # Enforce 12–12 window
    new_df = apply_12_12(new_df)
    if not master_df.empty:
        master_df = apply_12_12(master_df)

    # 3. DETECT OVERLAP (RED)
    if not master_df.empty:
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

    # 5. SORT & DEDUP (KEEP MASTER)
    combined_df = combined_df.sort_values("Timestamp")
    combined_df = combined_df.drop_duplicates(
        subset=["Timestamp", "Sensor_Name"],
        keep="first"
    )

    combined_df = combined_df.reset_index(drop=True)

    # 6. SAVE MASTER
    combined_df.to_csv(MASTER_CSV, index=False)

    # 7. COLORED EXCEL (safe)
    def highlight_rows(row):
        if row["Status"] == "New":
            return ["background-color:#90EE90"] * len(row)
        if row["Status"] == "Overlap":
            return ["background-color:#FF7F7F"] * len(row)
        if row["Status"] == "Historical":
            return ["background-color:#FFFFFF"] * len(row)
        return [""] * len(row)

    try:
        import openpyxl
        combined_df.style.apply(highlight_rows, axis=1)\
            .to_excel(MASTER_EXCEL, index=False, engine="openpyxl")
    except Exception:
        pass

    os.remove(filepath)

    st.success("CSV merged using Flask logic (12–12 enforced)")
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

if os.path.exists(MASTER_EXCEL):
    st.download_button(
        "Download Colored Excel",
        open(MASTER_EXCEL, "rb"),
        file_name="master_dataset_colored.xlsx"
    )

if st.button("Reset Master Dataset"):
    if os.path.exists(MASTER_CSV):
        os.remove(MASTER_CSV)
    if os.path.exists(MASTER_EXCEL):
        os.remove(MASTER_EXCEL)
    st.success("Master dataset reset. Reload page.")
