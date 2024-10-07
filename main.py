import streamlit as st 

st.title("sTARA: space Tool for Automated Research Analysis 🧬")
with st.sidebar:
        st.title("sTARA 🧬")
        st.write('Making information on space biology accessible and understandable to all!')
        
st.write("sTARA is an innovative multidisciplinary analytics tool designed specifically for space biology research published in NASA’s Open Science Data Repository (OSDR) and beyond. This open-source tool was developed to empower users—researchers, students, and enthusiasts alike—to effectively interpret complex biological data, particularly procedural information and sequence data available in the OSDR. The primary challenge addressed by sTARA is the difficulty users face in processing and analyzing large datasets with limited memory resources. By utilizing real-time data warehousing, sTARA efficiently manages substantial amounts of sequence data. The tool features advanced visualizations such as interactive heat maps, K-means clustering graphs for sequence data, and word clouds and interactive bar graphs for categorical data. Additionally, sTARA incorporates web scraping to streamline the inclusion of essential information, thus reducing the burden on users to manually gather data. sTARA is crucial for advancing space biology research as it enables direct comparisons between different studies, enhancing the ability to derive meaningful insights from diverse datasets. Ultimately, sTARA promotes open science and accessibility, fostering collaboration and innovation in space biology research for all!")

st.write("# Contents")
st.write("## Main: The Home Page (Here)")
st.write("## Basic Information: Speak With A Chatbot")
st.write("## Compare and Contrast: Compare Observations Between Studies")
st.write("## Sequencing: Make Observations of and Comparisons Between Raw Sequence Files")