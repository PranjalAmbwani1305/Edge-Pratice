import os
import pandas as pd
import streamlit as st
from datetime import time

# -------------------------------------------------
# Configuration (same as Flask)
# -------------------------------------------------
UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"
MASTER_EXCEL = "master_dataset_colored.xlsx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

START_TIME = time(3, 0, 0)   # 03:00
END_TIME   = time(12, 0, 0)  # 12:00

st.set_page_config(page_title="Master Dataset (Streamlit)", layout="centered")
st.title("Master Dataset Manager (Flask Logic + Time Window)")

# -------------------------------------------------
# Summary (same logic as Flask)
# -------------------------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None
    try:
        df = pd.read_csv(MASTER_CSV)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        status_counts = (
            df["Status"].value_counts().to_dict()
            if "Status" in df.columns else {}
        )

        return {
            "total_rows": len(df),
            "start_time": df["Timestamp"].min(),
            "end_time": df["Timestamp"].max(),
            "breakdown": df["Sensor_Name"].value_counts().to_dict(),
            "status_counts": status_counts,
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
    st.write("Sensor Breakdown:")
    st.json(summary["breakdown"])
    st.write("Status Breakdown:")
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

    # 1. LOAD NEW DATA (same as Flask)
    new_df = pd.read_csv(filepath)
    new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], errors="coerce")
    new_df["Sensor_Name"] = new_df["Sensor_Name"].astype(str).str.strip()
    new_df["Status"] = "New"

    # 🔑 TIME-WINDOW LOGIC (ONLY ADDITION)
    new_df = new_df[
        (new_df["Timestamp"].dt.time >= START_TIME) &
        (new_df["Timestamp"].dt.time < END_TIME)
    ]

    # 2. LOAD MASTER DATA (same as Flask)
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"], errors="coerce")
        master_df["Sensor_Name"] = master_df["Sensor_Name"].astype(str).str.strip()
        master_df["Status"] = "Historical"
    else:
        master_df = pd.DataFrame()

    # 3. DETECT OVERLAP (same as Flask)
    if not master_df.empty and not new_df.empty:
        master_keys = set(zip(master_df["Timestamp"], master_df["Sensor_Name"]))
        new_keys = set(zip(new_df["Timestamp"], new_df["Sensor_Name"]))
        overlap_keys = master_keys.intersection(new_keys)

        def mark_overlap(row):
            if (row["Timestamp"], row["Sensor_Name"]) in overlap_keys:
                return "Overlap"
            return "Historical"

        master_df["Status"] = master_df.apply(mark_overlap, axis=1)

    # 4. MERGE (same as Flask)
    combined_df = pd.concat([master_df, new_df])

    # 5. SORT & DEDUP (MASTER WINS – same as Flask)
    combined_df = combined_df.sort_values(by="Timestamp")
    combined_df = combined_df.drop_duplicates(
        subset=["Timestamp", "Sensor_Name"],
        keep="first"
    ).reset_index(drop=True)

    # 6. SAVE MASTER
    combined_df.to_csv(MASTER_CSV, index=False)

    # 7. GENERATE COLORED EXCEL (same as Flask)
    def highlight_rows(row):
        if row["Status"] == "New":
            return ["background-color: #90EE90"] * len(row)
        elif row["Status"] == "Overlap":
            return ["background-color: #FF7F7F"] * len(row)
        return ["background-color: #FFFFFF"] * len(row)

    try:
        import openpyxl
        combined_df.style.apply(highlight_rows, axis=1)\
            .to_excel(MASTER_EXCEL, index=False, engine="openpyxl")
    except Exception:
        combined_df.to_excel(MASTER_EXCEL, index=False)

    os.remove(filepath)

    st.success("CSV merged successfully (03:00 → 12:00 window applied)")
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
    st.success("Master dataset reset. Reload the page.")
