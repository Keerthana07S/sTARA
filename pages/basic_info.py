import streamlit as st
import openai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

#streamlit chat for improved UI
from streamlit_chat import message

st.set_page_config(page_title="Web Scraping Chatbot", page_icon="🤖")

#sidebar
with st.sidebar:
    st.title("NASA OSDR Scraper & Chatbot")
    st.write("This tool fetches content from a given URL, summarizes it, and interacts with you.")
    st.markdown("---")
    st.write("Enter a NASA OSDR URL to scrape data, and ask questions!")

#api key here
openai.api_key = 

#generate response from LLM by inputting prompt
def generate_response(prompt):
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Answer question on {prompt} based on information from {scraped_text}. {scraped_text} may be referred to as the 'study' or 'text' by the user.",
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].text.strip()

#selenium web driver for scraping
driver = webdriver.Chrome(ChromeDriverManager().install())

#title and url input
st.title("NASA OSDR URL Scraper")
st.write("### Paste a URL from NASA OSDR for analysis:")
url = st.text_input("Enter the NASA OSDR URL here:")

#initializing session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please enter a NASA OSDR URL and ask any questions you may have."}]

#chat interface rendering
def display_chat_interface():
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])
        else:
            with st.chat_message("user"):
                st.write(message["content"])

#display chat history
display_chat_interface()

#web scraping function
def scrape_nasa_osdr(url):
    try:
        wait = WebDriverWait(driver, 10)
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.ID, "description")))  # Assuming description element has ID "description"
        description = driver.find_element(By.ID, "description").text
        return description
    except Exception as e:
        return f"An error occurred during web scraping: {e}"

#main logic for url processing
if url:
    scraped_text = scrape_nasa_osdr(url)
    if scraped_text:
        #append the scraped text to the chat history but also summarize it
        with st.chat_message("assistant"):
            st.write("Scraped content from the URL:")
            st.write(scraped_text)

        #generate the summary using OpenAI
        prompt = f"Summarize the following NASA study text so that those who are not scientists can understand: {scraped_text}"
        summary = generate_response(prompt)

        #display the summary in the chat
        st.session_state.messages.append({"role": "assistant", "content": f"Here is the summarized content:\n\n{summary}"})

        with st.chat_message("assistant"):
            st.write("Summary:")
            st.write(summary)
else:
    with st.chat_message("assistant"):
        st.write("Please enter a valid NASA OSDR URL to fetch and summarize the content.")

#handling user input for the chatbot
if prompt := st.chat_input("Ask your question here..."):
    #append user input to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    #generate response using OpenAI
    response = generate_response(prompt)
    
    #append the assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

#clear chat history option in the sidebar
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
