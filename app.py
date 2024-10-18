# Bring in Dependencies
import getpass
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain  # Updated import
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

# Attempt to import API_KEY from api_key.py; fallback to environment variable
try:
    from api_key import API_KEY
except ImportError:
    API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    API_KEY = getpass.getpass("Enter your Groq API key: ")

os.environ['GROQ_API_KEY'] = API_KEY
# App Framework
st.title("LangChain YT Creator")
prompt = st.text_input("Enter your prompt here:")

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write me a YouTube video title about "{topic}".'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write me a YouTube video script based on this title : "{title}" also while leveraging the following research: {wikipedia_research}.'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define LLMChains with output_key
title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key='title',  # Specify the output key
    memory=title_memory  # Add memory to the chain
)

script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key='script',  # Specify the output key
    memory=script_memory  # Add memory to the chain
)

wiki = WikipediaAPIWrapper()

if prompt:
    # Run the chain with a dictionary input
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)

    # Display the results
    st.write("### Generated YouTube Title and Description")
    st.write(title)

    st.write("### Generated YouTube Script")
    st.write(script)

    with st.expander("Title History"):
        st.info(title_memory.buffer)
    
    with st.expander("Script History"):
        st.info(script_memory.buffer)
    
    with st.expander("Wikipedia Research History"):
        st.info(wiki_research)