import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# App Configuration
# -------------------------------------------------
st.set_page_config(page_title="CSV Master Merge", layout="centered")
st.title("Master Dataset Merge (Concat + De-Duplicate)")

MASTER_FILE = "master_dataset.csv"

# -------------------------------------------------
# Load or create master dataset
# -------------------------------------------------
if os.path.exists(MASTER_FILE):
    master_df = pd.read_csv(MASTER_FILE, parse_dates=["Timestamp"])
else:
    master_df = pd.DataFrame(
        columns=["Timestamp", "Sensor_Name", "Value"]
    )

st.info(f"Current records in master: {len(master_df)}")

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])

        # -----------------------------
        # Validate schema
        # -----------------------------
        required_cols = {"Timestamp", "Sensor_Name", "Value"}
        if not required_cols.issubset(new_df.columns):
            st.error("CSV must contain: Timestamp, Sensor_Name, Value")
        else:
            new_df["Sensor_Name"] = new_df["Sensor_Name"].astype(str).str.strip()

            st.write(
                f"Uploaded range: {new_df['Timestamp'].min()} → {new_df['Timestamp'].max()}"
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
            # REMOVE DUPLICATES (Uploaded wins)
            # -------------------------------------------------
            combined_df = combined_df.drop_duplicates(
                subset=["Timestamp", "Sensor_Name"],
                keep="last"   # uploaded CSV wins
            )

            # -------------------------------------------------
            # SORT ASCENDING BY TIMESTAMP
            # -------------------------------------------------
            combined_df.sort_values(
                by="Timestamp",
                ascending=True,
                inplace=True
            )

            # Save master
            combined_df.to_csv(MASTER_FILE, index=False)

            added_or_replaced = len(combined_df) - before_count

            st.success("CSV merged successfully")
            st.write("Rows added / replaced:", added_or_replaced)
            st.write("Total rows in master:", len(combined_df))

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
