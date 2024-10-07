import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
import traceback
import openai
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2

openai.api_key = ""

st.title("Gene Sequence Clustering Visualization")
with st.sidebar:
        st.title("sTARA 🧬")
        st.write("Making information on space biology understandable and accessible to all!")

        st.write("# Contents")
        st.write("## Main: The Homepage")
        st.write("## Basic Information: Speak With a Chatbot")
        st.write("## Compare and Contrast: Compare Observations Between Studies")
        st.write("## Sequencing: Make Observations of and Comparisions Between Raw Sequence Files (here)")
        st.write("## Quality Scores: Evaluate the Qualities of Gene Samples")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        st.write("Attempting to read the uploaded file...")
        pd_data = pd.read_csv(uploaded_file, nrows=5000) 

        pd_data.columns = pd_data.columns.str.strip().str.lower()

        st.write("Raw Column Names:", pd_data.columns.tolist())
        
        if 'sequence' not in pd_data.columns or 'seqid' not in pd_data.columns or 'quality' not in pd_data.columns:
            st.error("The uploaded CSV file must contain 'sequence', 'seqid', and 'quality' columns.")
        else:
            pd_data['sequence'] = pd_data['sequence'].astype(str).fillna('')
            pd_data['seqid'] = pd_data['seqid'].astype(str).fillna('')
            pd_data['quality'] = pd_data['quality'].astype(str).fillna('')

            st.write("Data Loaded Successfully!")

            pd_data['quality_numeric'] = pd_data['quality'].str.count('F')

            def kmer_encoding(sequence, k=3, max_length=200):
                if pd.isna(sequence):
                    return {}
                sequence = sequence[:max_length]  # Truncate to max_length
                return {sequence[i:i+k]: 1 for i in range(len(sequence) - k + 1) if len(sequence) >= k}

            def batch_kmer_encoding(pandas_series, batch_size=50, k=3, max_length=200):
                all_kmers = []
                num_batches = len(pandas_series) // batch_size + 1
                for i in range(num_batches):
                    batch_df = pandas_series[i*batch_size:(i+1)*batch_size]
                    kmers_batch = [kmer_encoding(seq, k=k, max_length=max_length) for seq in batch_df]
                    all_kmers.append(pd.DataFrame(kmers_batch).fillna(0))
                return pd.concat(all_kmers, ignore_index=True)

            kmer_df = batch_kmer_encoding(pd_data['sequence'], batch_size=50, k=3, max_length=200)

            kmer_df = kmer_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            features = pd.concat([kmer_df, pd_data[['quality_numeric']]], axis=1)

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features).astype(np.float32)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_features)

            num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters)
            clusters = kmeans.fit_predict(pca_result)

            silhouette_avg = silhouette_score(pca_result, clusters)
            davies_bouldin_avg = davies_bouldin_score(pca_result, clusters)
            st.write(f'Silhouette Score: {silhouette_avg:.2f}')
            st.write(f'Davies-Bouldin Score: {davies_bouldin_avg:.2f}')

            pd_data['sequence_length'] = pd_data['sequence'].apply(len)
            pd_data['gc_content'] = pd_data['sequence'].apply(lambda x: (x.count('G') + x.count('C')) / len(x) * 100 if len(x) > 0 else 0)

            cluster_labels = [str(label) for label in clusters] 
            source = ColumnDataSource(data={
                'x': pca_result[:, 0],
                'y': pca_result[:, 1],
                'cluster': cluster_labels,
                'seqid': pd_data['seqid'].tolist()
            })

            color_map = factor_cmap('cluster', palette=Category10[num_clusters], factors=sorted(set(cluster_labels)))

            plot = figure(title="K-Means Clustering of Gene Sequences",
                        x_axis_label='PCA Component 1',
                        y_axis_label='PCA Component 2')

            plot.scatter('x', 'y', source=source, fill_alpha=0.6, size=8, color=color_map, legend_field='cluster')

            kmer_counts = kmer_df.sum(axis=0)
            top_kmers = kmer_counts.nlargest(10) 

            plot.legend.title = 'Clusters'
            plot.legend.location = "top_left"
            plot.legend.click_policy = "hide"

            plot.add_tools(HoverTool(tooltips=[("SeqID", "@seqid"), ("Cluster", "@cluster")])) 

            st.bokeh_chart(plot)

        species = st.text_input("Enter the species name:")

        unique_kmers_count = len(kmer_counts)
        sequence_length_distribution = pd_data['sequence_length'].describe()
        gc_content_distribution = pd_data['gc_content'].describe()
        unique_variations_count = pd_data['sequence'].nunique()
        
        unique_sequences = pd_data['sequence'].nunique()
        common_motifs = pd_data['sequence'].str.extract(r'(?=(\w{3}))')[0].value_counts().nlargest(5).to_dict()  
        variations = pd_data['sequence'].str.extractall(r'(\w)')
        unique_variations_count = pd_data['sequence'].nunique()

        pd_data['gc_content'] = pd_data['sequence'].apply(lambda x: (x.count('G') + x.count('C')) / len(x) * 100 if len(x) > 0 else 0)

        report_variables = {
            "Species": species,
            "Sequence Length Distribution (Mean, Median)": {
                "Mean": pd_data['sequence'].str.len().mean(),
                "Median": pd_data['sequence'].str.len().median()
            },
            "GC Content Distribution (Mean, Range)": {
                "Mean": pd_data['gc_content'].mean(),
                "Min": pd_data['gc_content'].min(),
                "Max": pd_data['gc_content'].max()
            },
            "Unique Sequence Count": pd_data['sequence'].nunique(),
            "Common Motifs Count": pd_data['sequence'].str.extract(r'(?=(\w{3}))')[0].value_counts().nlargest(5).count(),
            "Variations Count": pd_data['sequence'].nunique() 
        }


        response = openai.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates detailed scientific reports."},
                {"role": "user", "content": f"Generate a report based on the following data, be detailed about what the variables mean and give an overall statement of the quality of the sample: {report_variables}"}
            ]
        )

        st.write(response['choices'][0]['message']['content'])

        
    except Exception as e:
        st.error(f"Error generating report: {e}")
    except MemoryError as mem_err:
        st.error(f"MemoryError: {mem_err}. Consider reducing the data size or processing less data.")
    except pd.errors.ParserError as parse_err:
        st.error(f"Error parsing the CSV file: {parse_err}")
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")
        st.error(traceback.format_exc())
else:
    st.warning("Please upload a CSV file.")
