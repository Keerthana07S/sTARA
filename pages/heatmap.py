import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Function to convert ASCII to Phred+33 quality scores
def phred_quality(quality_string):
    return [ord(char) - 33 for char in quality_string]

# Streamlit app
st.title("Quality Score Heatmap Generator")
with st.sidebar:
    st.title("sTARA 🧬")
    st.write("Making information on space biology understandable and accessible to all!")

    st.write("# Contents")
    st.write("## Main: The Homepage")
    st.write("## Basic Information: Speak With a Chatbot")
    st.write("## Compare and Contrast: Compare Observations Between Studies")
    st.write("## Sequencing: Make Observations of and Comparisons Between Raw Sequence Files")
    st.write("## Quality Scores: Evaluate the Qualities of Gene Samples (here)")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data 
    def load_data(file):
        return pd.read_csv(file)

    data = load_data(uploaded_file)

    if data.empty:
        st.error("The uploaded file is empty or not formatted correctly.")
    else:
        data['average_quality'] = data['quality'].str.encode('ascii').apply(lambda x: sum(phred_quality(x.decode())) / len(x))

        num_samples = len(data)
        samples_per_page = 25
        num_pages = (num_samples + samples_per_page - 1) // samples_per_page  # Fixed page calculation

        page = st.selectbox("Select Page", range(num_pages))

        start_index = page * samples_per_page
        end_index = min(start_index + samples_per_page, num_samples)  # Ensure it doesn't exceed the total samples
        
        heatmap_data = data.iloc[start_index:end_index].pivot_table(index='SeqID', values='average_quality', aggfunc='mean')

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='viridis', annot=False)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        plt.title("Average Quality Score per Sequence", fontsize=14)
        plt.xlabel("Average Quality Score", fontsize=12)
        plt.ylabel("Sequence ID", fontsize=12)

        st.pyplot(plt)
