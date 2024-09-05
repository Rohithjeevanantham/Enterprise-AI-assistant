import streamlit as st
import pandas as pd
from ragpart import generate_response_from_chunks, get_relevant_chunks, create_index, extract_text_from_pdf, clean_text, store_chunks_in_pinecone, combined_chunking
from translate import translate, generate_audio

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'query' not in st.session_state:
    st.session_state.query = None
if 'papers_downloaded' not in st.session_state:
    st.session_state.papers_downloaded = False

def reset_page():
    st.session_state.index = None
    st.session_state.query = None
    st.session_state.papers_downloaded = False

# Streamlit app
st.sidebar.image("ai_icon.png")
st.title("Araycci Research Paper Bot")
st.sidebar.title("PDF Research Assistant")

lang = st.sidebar.radio("Choose", ["English", "French", "Spanish"])

# Language map
language_map = {
    'English': 'en-US',
    'Spanish': 'es-ES',
    'French': 'fr-FR'
}

def process_local_pdfs(data):
    combined_chunks = []
    
    # Check if data is a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_dict()
        data = data['text']

    # If data is a list of uploaded files
    for pdf_file in data:
        if isinstance(data, dict) and isinstance(data[pdf_file], str):
            text = data[pdf_file]  
        else:
            text = extract_text_from_pdf(pdf_file)
        
        cleaned_text = clean_text(text)
        chunks = combined_chunking(cleaned_text)
        combined_chunks.extend(chunks)
    
    return combined_chunks

def handle_query_response(query, lang):
    relevant_chunks = get_relevant_chunks(query, st.session_state.index)
    response = generate_response_from_chunks(relevant_chunks, query)
    if lang != "English":
        translated_response = translate(response, lang)
        st.write(translated_response)
        audio_io = generate_audio(translated_response, lang)
    else:
        st.write(response)
        audio_io = generate_audio(response, lang)
    st.audio(audio_io, format='audio/mp3')
    st.download_button(label="Download Audio Response", data=audio_io, file_name="response.mp3", mime="audio/mp3")

# Handle Local PDF Processing
data = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
if data and not st.session_state.papers_downloaded:
    with st.spinner("Processing PDFs..."):
        combined_chunks = process_local_pdfs(data)
        st.session_state.index = create_index()
        if st.session_state.index:
            store_chunks_in_pinecone(combined_chunks, st.session_state.index)
            st.session_state.papers_downloaded = True
            st.success("PDF processed and indexed successfully!")
        else:
            st.error("Failed to create Pinecone index.")

# Query handling
if st.session_state.index:
    query = st.text_input("Enter your question:")
    if query:
        st.session_state.query = query
    if st.button("Ask") and st.session_state.query:
        with st.spinner("Searching for answers..."):
            handle_query_response(st.session_state.query, lang)
        
    if st.button("End conversation"):
        reset_page()
        st.rerun()
