import openai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By 
import codecs
import streamlit.components.v1 as components
import re
from webdriver_manager.chrome import ChromeDriverManager
import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import io
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math import pi
from matplotlib_venn import venn2

driver = webdriver.Chrome(ChromeDriverManager().install())
openai.api_key = ""

panel_text = 0
panel_text2 = 0
data1 = ""
data2 = ""
common_columns = ""

st.title("Compare and Contrast 🔎")
st.write("This page is intended to support quantitative and qualitative observations of and comparisons between academic research.")
url = st.text_input("Enter the URL for Text 1")
url2 = st.text_input("Enter a second URL Text 2")

def read_csv(uploaded_file):
    data = []
    content = uploaded_file.read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        data.append(row)
    return data

def identify_categorical_columns(data):
    #identify categorical columns (assuming dtype 'object' indicates categorical)
    categorical_columns = data.select_dtypes(include=['object']).columns
    return categorical_columns

def create_comparison_bar_chart(count1, count2, category1, category2):
    #set up bar chart data
    labels = list(set(count1.index).union(set(count2.index)))
    values1 = [count1.get(label, 0) for label in labels]
    values2 = [count2.get(label, 0) for label in labels]

    x = range(len(labels))  #the label locations

    fig, ax = plt.subplots()
    bars1 = ax.bar(x, values1, width=0.4, label=category1, align='center')
    bars2 = ax.bar(x, values2, width=0.4, label=category2, align='edge')

    #add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_title(f'Comparison of {category1} and {category2}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    #display the plot in Streamlit
    st.pyplot(fig)

def display_category_comparison(df1, df2):
    #get unique categories from both DataFrames
    categories1 = df1.columns.tolist()
    categories2 = df2.columns.tolist()

    st.write("Select categories to compare from each dataset:")

    #dropdowns for selecting categories
    selected_category1 = st.selectbox("Select a category from Study 1", categories1)
    selected_category2 = st.selectbox("Select a category from Study 2", categories2)

    #display the counts for the selected categories
    if selected_category1 in df1.columns and selected_category2 in df2.columns:
        count1 = df1[selected_category1].value_counts()
        count2 = df2[selected_category2].value_counts()

        #display counts
        st.write(f"Counts for {selected_category1} in Study 1:")
        st.write(count1)

        st.write(f"Counts for {selected_category2} in Study 2:")
        st.write(count2)

        #create a bar chart to visualize the counts
        create_comparison_bar_chart(count1, count2, selected_category1, selected_category2)

def display_basic_stats(data):
    st.write("Basic Dataset Statistics:")
    st.write("Number of Rows:", len(data))
    st.write("Number of Columns:", len(data.columns))
    
    #missing Values
    st.write("Missing Values:")
    st.write(data.isnull().sum())
    
    #descriptive Summary of Numeric Columns
    st.write("Descriptive Summary of Numeric Columns:")
    st.write(data.describe())
    
def display_descriptive_summary(data1, data2, common_numeric_columns):
    st.write("### Descriptive Summary of Numeric Columns:")
    for col in common_numeric_columns:
        st.write(f"**{col}**")
        summary1 = data1[col].describe()
        summary2 = data2[col].describe()

        col1, col2 = st.columns(2)
        col1.write(f"Study 1: {summary1}")
        col2.write(f"Study 2: {summary2}")
        
def compare_common_columns(data1, data2):
    st.write("### Side-by-Side Data Comparison:")
    common_columns = list(set(data1.columns).intersection(set(data2.columns)))
    selected_columns = st.multiselect("Select common columns to display", common_columns, default=common_columns)

    if len(selected_columns) > 0:
        col1, col2 = st.columns(2)
        col1.write("Study 1 Data")
        col1.write(data1[selected_columns].head())
        
        col2.write("Study 2 Data")
        col2.write(data2[selected_columns].head())
        
def display_wordcloud(text1, text2):
    st.write("### Word Clouds for Text Comparison")
    col1, col2 = st.columns(2)
    
    wordcloud1 = WordCloud(width=400, height=200).generate(text1)
    wordcloud2 = WordCloud(width=400, height=200).generate(text2)
    
    col1.write("Text 1 Word Cloud")
    col1.image(wordcloud1.to_array())

    col2.write("Text 2 Word Cloud")
    col2.image(wordcloud2.to_array())

def display_category_counts(df1, df2):
    st.write("### Bar Chart Comparison of Categorical Counts")
    categories1 = df1['category'].value_counts()
    categories2 = df2['category'].value_counts()
    
    data = {
        'Category': list(set(categories1.index).union(set(categories2.index))),
        'Count1': [categories1.get(cat, 0) for cat in categories1.index.union(categories2.index)],
        'Count2': [categories2.get(cat, 0) for cat in categories2.index.union(categories1.index)],
    }
    df_counts = pd.DataFrame(data)
    
    df_counts.plot(kind='bar', x='Category', y=['Count1', 'Count2'], color=['blue', 'orange'])
    plt.title("Comparison of Categorical Counts")
    st.pyplot(plt)
    

def display_text_similarity_heatmap(text1, text2):
    st.write("### Text Similarity Heatmap")
    texts = [text1, text2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", xticklabels=["Text 1", "Text 2"], yticklabels=["Text 1", "Text 2"])
    plt.title("Cosine Similarity Heatmap")
    st.pyplot(plt)

def display_radar_chart(df1, df2):
    st.write("### Radar Chart Comparison of Themes")
    
    themes = ['Theme1', 'Theme2', 'Theme3']  # Replace with actual theme names
    scores1 = [df1[theme].mean() for theme in themes]
    scores2 = [df2[theme].mean() for theme in themes]
    
    N = len(themes)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    scores1 += scores1[:1]
    scores2 += scores2[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], themes)
    ax.plot(angles, scores1, linewidth=2, linestyle='solid', label="Dataset1")
    ax.fill(angles, scores1, 'b', alpha=0.1)
    ax.plot(angles, scores2, linewidth=2, linestyle='solid', label="Dataset2")
    ax.fill(angles, scores2, 'r', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Radar Chart Comparison of Themes")
    st.pyplot(plt)


def extract_segment(text, start_word, end_word=None):
    start_index = text.find(start_word)
    if start_index == -1:
        return None
    if end_word:
        end_index = text.find(end_word, start_index + len(start_word))
        if end_index != -1 and start_index < end_index:
            return text[start_index + len(start_word):end_index].strip()
    else:
        return text[start_index + len(start_word):].strip()
    return None

def mermaid(code: str) -> None:
            components.html(
                f"""
                <div>
                    <pre class="mermaid">
                        {code}
                    </pre>
                    <button id="downloadSvgBtn">Download as SVG</button>
                    <button id="downloadPngBtn">Download as PNG</button>
                </div>

                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true }});

                    document.getElementById('downloadSvgBtn').addEventListener('click', function() {{
                        const svg = document.querySelector('.mermaid svg');
                        const svgData = new XMLSerializer().serializeToString(svg);
                        const blob = new Blob([svgData], {{ type: 'image/svg+xml' }});
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'mermaid_diagram.svg';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }});

                    document.getElementById('downloadPngBtn').addEventListener('click', function() {{
                        const svg = document.querySelector('.mermaid svg');
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');
                        const svgData = new XMLSerializer().serializeToString(svg);
                        const img = new Image();
                        img.onload = function() {{
                            context.drawImage(img, 0, 0);
                            const pngData = canvas.toDataURL('image/png');
                            const downloadLink = document.createElement('a');
                            downloadLink.href = pngData;
                            downloadLink.download = 'mermaid_diagram.png';
                            downloadLink.click();
                        }};
                        img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
                    }});
                </script>
                """,
                height=2000
            )

st.session_state.messages = [{"role": "assistant"}]


if url or url2 is None:
    st.write("Compare between two studies.")

if url and url2 is not None:
    wait = WebDriverWait(driver, 10)
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.ID, "description")))
    expansion_panel = driver.find_element(By.ID, "description")
    panel_text = expansion_panel.text
    
    wait = WebDriverWait(driver, 10)
    driver.get(url2)
    wait.until(EC.presence_of_element_located((By.ID, "description")))
    expansion_panel2 = driver.find_element(By.ID, "description")
    panel_text2 = expansion_panel2.text
    chat_prompt = {
                                "role": "user",
                                "content": f"Please based on the user input, write the Mermaid code for the required process. User input:\n {panel_text}\n"
                    }
    with st.spinner("Thinking..."):
        response = openai.completions.create(model = "gpt-3.5-turbo-instruct", prompt = chat_prompt, max_tokens=150, temperature=0.5)
        text = response.choices[0].text
        message = {"content": text}
        st.session_state.messages.append(message) 
        st.session_state.merm_code = extract_segment(text, "mermaid", "END")

        if st.session_state.merm_code is not None:
            mermaid(
                f"""
                {st.session_state.merm_code}
                """
            )
    response = openai.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=(f"Please summarize the following texts and explain why they are both different. Text 1: {panel_text}. Text 2: {panel_text2}. Refer to each as 'Text 1' and 'Text 2' as needed"),
    max_tokens=150,
    temperature=0.5
    )
    summary = response.choices[0].text
    st.write(summary)
    
    display_wordcloud(panel_text, panel_text2)
    
    


col1, col2 = st.columns([10,10])


uploaded_file_1 = st.file_uploader("Choose a CSV file for any study", type=["csv"])
uploaded_file_2 = st.file_uploader("Choose a CSV file for another study", type=["csv"])

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    data1 = pd.read_csv(uploaded_file_1)
    data2 = pd.read_csv(uploaded_file_2)

    with col1:
        st.write("## Study 1 Data Description")
        display_basic_stats(data1)
    
    with col2:
        st.write("## Study 2 Data Description")
        display_basic_stats(data2)

    common_columns = list(set(data1.columns).intersection(set(data2.columns)))
    numeric_columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(data1[col])]

    display_descriptive_summary(data1, data2, numeric_columns)
    display_category_comparison(data1, data2)
    compare_common_columns(data1, data2)
