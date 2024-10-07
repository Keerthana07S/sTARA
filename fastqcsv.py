import streamlit as st
from snowflake.snowpark import Session
import snowflake.connector
import pandas as pd
from Bio import SeqIO
import gzip
import tempfile

# Snowflake connection configuration
snowflake_conn = snowflake.connector.connect(
    user="username",
    password="password",
    account="account identification",
    role="account type",
    warehouse="warehouse name",
    database="database name",
    schema="schema",
)

# File uploader for FASTQ files
uploaded_file = st.file_uploader("Choose a FASTQ file", type=["fastq.gz"])

# Function to convert FASTQ to CSV and upload to Snowflake
def fastq_to_csv(fastq_file):
    try:
        # Use gzip to handle compressed FASTQ files
        with gzip.open(fastq_file, "rt") as handle:
            records = []
            for record in SeqIO.parse(handle, "fastq"):
                records.append({
                    "ID": record.id,
                    "Sequence": str(record.seq),
                    "Quality": ",".join(map(str, record.letter_annotations["phred_quality"]))
                })
        # Create a DataFrame from the records
        df = pd.DataFrame(records)

        # Save the DataFrame to a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            csv_file = temp_file.name
            df.to_csv(csv_file, index=False)

        # Upload CSV to Snowflake external AWS stage using the Snowflake Connector
        cursor = snowflake_conn.cursor()
        try:
            # Use the PUT command to upload the file to the external stage
            cursor.execute(f"PUT file://{csv_file} @spaceappsbio AUTO_COMPRESS=true")
        finally:
            cursor.close()  # Ensure the cursor is closed after the operation

        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Process the uploaded file if it exists
if uploaded_file is not None:
    # Store the uploaded file in a temporary location
    with tempfile.NamedTemporaryFile(suffix=".fastq.gz", delete=False) as temp_fastq:
        temp_fastq.write(uploaded_file.read())
        temp_fastq_path = temp_fastq.name

    # Convert FASTQ to CSV and preview it
    df = fastq_to_csv(temp_fastq_path)
    if df is not None:
        st.write("Preview of the converted CSV file:")
        st.dataframe(df.head())
        st.write("Basic Statistics:")
        st.write(f"Total Records: {len(df)}")
        st.write(f"Mean Sequence Length: {df['Sequence'].apply(len).mean():.2f}")
        st.write(f"Mean Quality Score: {df['Quality'].apply(lambda x: sum(map(int, x.split(','))) / len(x.split(','))).mean():.2f}")
    else:
        st.error("Failed to process the uploaded FASTQ file.")
