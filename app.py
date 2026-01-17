import os
import pandas as pd
import streamlit as st

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(page_title="CSV Master Merge", layout="centered")
st.title("Master Dataset Manager (Streamlit Version)")

UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"
MASTER_EXCEL = "master_dataset_colored.xlsx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------
# Utility: Load Summary
# ---------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None
    try:
        df = pd.read_csv(MASTER_CSV)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        status_counts = (
            df["Status"].value_counts().to_dict()
            if "Status" in df.columns
            else {}
        )

        summary = {
            "total_rows": len(df),
            "start_time": df["Timestamp"].min(),
            "end_time": df["Timestamp"].max(),
            "breakdown": df["Sensor_Name"].value_counts().to_dict(),
            "status_counts": status_counts,
        }
        return summary
    except Exception as e:
        st.error(f"Error reading summary: {e}")
        return None


# ---------------------------------
# Show Summary
# ---------------------------------
summary = get_summary()

if summary:
    st.subheader("Master Dataset Summary")
    st.write(f"Total Rows: {summary['total_rows']}")
    st.write(f"Start Time: {summary['start_time']}")
    st.write(f"End Time: {summary['end_time']}")

    st.write("Sensor Breakdown:")
    st.json(summary["breakdown"])

    st.write("Status Breakdown:")
    st.json(summary["status_counts"])
else:
    st.info("No master dataset found yet.")

st.divider()

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Load NEW data (Green)
    try:
        new_df = pd.read_csv(filepath)
        new_df["Timestamp"] = pd.to_datetime(new_df["Timestamp"], errors="coerce")
        new_df["Sensor_Name"] = new_df["Sensor_Name"].astype(str).str.strip()
        new_df["Status"] = "New"
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # 2. Load MASTER data (White / Red)
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df["Timestamp"] = pd.to_datetime(master_df["Timestamp"], errors="coerce")
        master_df["Sensor_Name"] = master_df["Sensor_Name"].astype(str).str.strip()
        master_df["Status"] = "Historical"
    else:
        master_df = pd.DataFrame()

    # 3. Detect Overlap (Red)
    if not master_df.empty and not new_df.empty:
        master_keys = set(zip(master_df["Timestamp"], master_df["Sensor_Name"]))
        new_keys = set(zip(new_df["Timestamp"], new_df["Sensor_Name"]))
        overlap_keys = master_keys.intersection(new_keys)

        def mark_overlap(row):
            if (row["Timestamp"], row["Sensor_Name"]) in overlap_keys:
                return "Overlap"
            return "Historical"

        master_df["Status"] = master_df.apply(mark_overlap, axis=1)

    # 4. Merge
    combined_df = pd.concat([master_df, new_df], ignore_index=True)

    # 5. Sort & Deduplicate (KEEP MASTER ON OVERLAP)
    combined_df = combined_df.sort_values(by="Timestamp")
    combined_df = combined_df.drop_duplicates(
        subset=["Timestamp", "Sensor_Name"],
        keep="first"
    )

    # 6. Reset index (Excel-safe)
    combined_df = combined_df.reset_index(drop=True)

    # 7. Save CSV
    combined_df.to_csv(MASTER_CSV, index=False)

    # 8. Generate Color-Coded Excel
    def highlight_rows(row):
        status = row["Status"]
        if status == "New":
            return ["background-color: #90EE90"] * len(row)
        elif status == "Overlap":
            return ["background-color: #FF7F7F"] * len(row)
        elif status == "Historical":
            return ["background-color: #FFFFFF"] * len(row)
        return [""] * len(row)

    try:
        styled_df = combined_df.style.apply(highlight_rows, axis=1)
        styled_df.to_excel(MASTER_EXCEL, index=False, engine="openpyxl")
    except Exception:
        combined_df.to_excel(MASTER_EXCEL, index=False)

    os.remove(filepath)

    st.success("CSV uploaded and master dataset updated!")

    st.subheader("Updated Master Preview")
    st.dataframe(combined_df.tail(20))

# ---------------------------------
# Downloads
# ---------------------------------
st.divider()

if os.path.exists(MASTER_EXCEL):
    with open(MASTER_EXCEL, "rb") as f:
        st.download_button(
            "Download Colored Excel",
            f,
            file_name=MASTER_EXCEL,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

if os.path.exists(MASTER_CSV):
    with open(MASTER_CSV, "rb") as f:
        st.download_button(
            "Download Master CSV",
            f,
            file_name=MASTER_CSV,
            mime="text/csv",
        )

# ---------------------------------
# Reset
# ---------------------------------
if st.button("Reset Master Dataset"):
    if os.path.exists(MASTER_CSV):
        os.remove(MASTER_CSV)
    if os.path.exists(MASTER_EXCEL):
        os.remove(MASTER_EXCEL)
    st.success("Master dataset reset. Reload the page.")
