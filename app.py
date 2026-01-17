import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# App Configuration
# -------------------------------------------------
st.set_page_config(page_title="Edge CSV Merge", layout="centered")
st.title("Edge Sensor CSV Merge (Master Dataset)")

MASTER_FILE = "master_dataset.csv"

# -------------------------------------------------
# Utility: normalize column names
# -------------------------------------------------
def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# -------------------------------------------------
# Load or create master dataset
# -------------------------------------------------
if os.path.exists(MASTER_FILE):
    master_df = pd.read_csv(MASTER_FILE)
    master_df = normalize_columns(master_df)
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value"]
    )

# Ensure timestamp is datetime
master_df["timestamp"] = pd.to_datetime(
    master_df["timestamp"], errors="coerce"
)

st.info(f"Current records in master: {len(master_df)}")

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Drag and drop CSV file here",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        new_df = normalize_columns(new_df)

        # -------------------------------------------------
        # Validate schema (FIXED)
        # -------------------------------------------------
        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(new_df.columns):
            st.error(
                "CSV must contain columns: timestamp, sensor, voltage, adc_value"
            )
        else:
            # Parse & clean
            new_df["timestamp"] = pd.to_datetime(
                new_df["timestamp"], errors="coerce"
            )
            new_df["sensor"] = new_df["sensor"].astype(str).str.strip()

            st.write(
                f"Uploaded range: {new_df['timestamp'].min()} → {new_df['timestamp'].max()}"
            )

            before_count = len(master_df)

            # -------------------------------------------------
            # CONCAT master + uploaded
            # -------------------------------------------------
            combined_df = pd.concat(
                [master_df, new_df],
                ignore_index=True
            )

            # -------------------------------------------------
            # REMOVE DUPLICATES (timestamp + sensor)
            # Uploaded CSV wins
            # -------------------------------------------------
            combined_df = combined_df.drop_duplicates(
                subset=["timestamp", "sensor"],
                keep="last"
            )

            # -------------------------------------------------
            # SORT ASCENDING BY TIMESTAMP
            # -------------------------------------------------
            combined_df.sort_values(
                by="timestamp",
                ascending=True,
                inplace=True
            )

            # Save master
            combined_df.to_csv(MASTER_FILE, index=False)

            added_or_replaced = len(combined_df) - before_count

            st.success("CSV merged successfully")
            st.write("Rows added / replaced:", added_or_replaced)
            st.write("Total records in master:", len(combined_df))

            st.subheader("Master Dataset Preview")
            st.dataframe(combined_df.tail(20))

            # Update in-memory master
            master_df = combined_df

    except Exception as e:
        st.error(f"Error processing file: {e}")

# -------------------------------------------------
# Download master dataset
# -------------------------------------------------
if os.path.exists(MASTER_FILE):
    with open(MASTER_FILE, "rb") as f:
        st.download_button(
            label="Download Master Dataset",
            data=f,
            file_name="master_dataset.csv",
            mime="text/csv"
        )
