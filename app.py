import os
import pandas as pd
import streamlit as st

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(page_title="Master Dataset Manager", layout="centered")
st.title("Master Dataset Manager (Streamlit)")

UPLOAD_FOLDER = "uploads"
MASTER_CSV = "master_dataset.csv"
MASTER_EXCEL = "master_dataset_colored.xlsx"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------
# Utility: Normalize column names
# ---------------------------------
def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ---------------------------------
# Utility: Load Summary
# ---------------------------------
def get_summary():
    if not os.path.exists(MASTER_CSV):
        return None
    try:
        df = pd.read_csv(MASTER_CSV)
        df = normalize_columns(df)

        # Required safety checks
        if "timestamp" not in df.columns or "sensor" not in df.columns:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        status_counts = (
            df["status"].value_counts().to_dict()
            if "status" in df.columns
            else {}
        )

        summary = {
            "total_rows": len(df),
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max(),
            "breakdown": df["sensor"].value_counts().to_dict(),
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
    "Drag and drop CSV file here",
    type=["csv"]
)

if uploaded_file:
    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Load NEW data (Green)
    try:
        new_df = pd.read_csv(filepath)
        new_df = normalize_columns(new_df)

        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(new_df.columns):
            st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
            st.stop()

        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], errors="coerce")
        new_df["sensor"] = new_df["sensor"].astype(str).str.strip()
        new_df["status"] = "New"

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # 2. Load MASTER data (White / Red)
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df = normalize_columns(master_df)

        master_df["timestamp"] = pd.to_datetime(master_df["timestamp"], errors="coerce")
        master_df["sensor"] = master_df["sensor"].astype(str).str.strip()
        master_df["status"] = "Historical"
    else:
        master_df = pd.DataFrame(columns=new_df.columns)

    # 3. Detect Overlap (Red)
    if not master_df.empty:
        master_keys = set(zip(master_df["timestamp"], master_df["sensor"]))
        new_keys = set(zip(new_df["timestamp"], new_df["sensor"]))
        overlap_keys = master_keys.intersection(new_keys)

        master_df.loc[
            master_df.apply(
                lambda r: (r["timestamp"], r["sensor"]) in overlap_keys, axis=1
            ),
            "status"
        ] = "Overlap"

    # 4. Merge
    combined_df = pd.concat([master_df, new_df], ignore_index=True)

    # 5. Sort & Deduplicate (KEEP MASTER ON DUPLICATE)
    combined_df = combined_df.sort_values(by="timestamp")
    combined_df = combined_df.drop_duplicates(
        subset=["timestamp", "sensor"],
        keep="first"
    )

    # 6. Reset index (Excel-safe)
    combined_df = combined_df.reset_index(drop=True)

    # 7. Save CSV
    combined_df.to_csv(MASTER_CSV, index=False)

    # 8. Generate Color-Coded Excel
    def highlight_rows(row):
        status = row.get("status", "")
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
    st.subheader("Updated Master Dataset Preview")
    st.dataframe(combined_df.tail(20))

st.divider()

# ---------------------------------
# Downloads
# ---------------------------------
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
