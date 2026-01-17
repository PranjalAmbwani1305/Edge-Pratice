import streamlit as st
import pandas as pd
import os

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="CSV Merge App", layout="centered")
st.title("Master Dataset Merge (Concat + Remove Duplicates)")

MASTER_FILE = "master_dataset.csv"

# ---------------------------------
# Load or create master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    master_df = pd.read_csv(MASTER_FILE, parse_dates=["timestamp"])
else:
    master_df = pd.DataFrame(
        columns=["timestamp", "sensor", "voltage", "adc_value"]
    )

st.write("Current records in master:", len(master_df))

# ---------------------------------
# Upload CSV
# ---------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        # Validate schema
        required_cols = {"timestamp", "sensor", "voltage", "adc_value"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain: timestamp, sensor, voltage, adc_value")
        else:
            st.info(
                f"Uploaded range: {df['timestamp'].min()} → {df['timestamp'].max()}"
            )

            before_count = len(master_df)

            # ---------------------------------
            # CONCAT master + uploaded
            # ---------------------------------
            combined_df = pd.concat(
                [master_df, df],
                ignore_index=True
            )

            # ---------------------------------
            # REMOVE DUPLICATES
            # (timestamp + sensor)
            # ---------------------------------
            combined_df.drop_duplicates(
                subset=["timestamp", "sensor"],
                keep="first",
                inplace=True
            )

            # ---------------------------------
            # SORT ASCENDING BY DATE
            # ---------------------------------
            combined_df.sort_values(
                by="timestamp",
                ascending=True,
                inplace=True
            )

            # Save master
            combined_df.to_csv(MASTER_FILE, index=False)

            added = len(combined_df) - before_count

            st.success("CSV merged successfully")
            st.write("New records added:", added)
            st.write("Total records in master:", len(combined_df))

            st.subheader("Master Dataset Preview")
            st.dataframe(combined_df.tail(15))

            # Update in-memory master
            master_df = combined_df

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---------------------------------
# Download master dataset
# ---------------------------------
if os.path.exists(MASTER_FILE):
    with open(MASTER_FILE, "rb") as f:
        st.download_button(
            label="Download Master Dataset",
            data=f,
            file_name="master_dataset.csv",
            mime="text/csv"
        )
