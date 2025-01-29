import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
import speech_recognition as sr
from fpdf import FPDF

# Disable GPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add template imports
sys.path.append("c:/Users/manue/Desktop/DataScience/PDF/pdf")
from templates import css, bot_template, user_template

def get_pdf_text(pdf):
    """Extracts text from a selected PDF."""
    reader = PdfReader(pdf)
    return "".join([page.extract_text() for page in reader.pages])

def get_text_chunks(text):
    """Splits text into chunks."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vectorstore(chunks):
    """Creates FAISS vectorstore from text chunks."""
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embedding)

def get_conversation_chain(vectorstore):
    """Initializes conversation model."""
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    """Handles user questions and updates chat UI."""
    typing_placeholder = st.empty()
    typing_placeholder.write('<div class="chat-message bot"><div class="message">Typing...</div></div>', unsafe_allow_html=True)
    time.sleep(1)
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    typing_placeholder.empty()
    
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def listen_for_question():
    """Function to use speech recognition to listen for a user question."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = r.listen(source)
        try:
            question = r.recognize_google(audio)
            st.write(f"Your question: {question}")
            return question
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand your question.")
            return ""
        except sr.RequestError:
            st.write("Sorry, I'm having trouble accessing the speech recognition service.")
            return ""

def export_chat_to_text():
    """Exports the chat history to a text file."""
    chat_text = "\n".join([message.content for message in st.session_state.chat_history])
    with open("chat_log.txt", "w") as f:
        f.write(chat_text)
    st.download_button("Download Chat as Text", "chat_log.txt")

def export_chat_to_pdf():
    """Exports the chat history to a PDF."""
    chat_pdf = FPDF()
    chat_pdf.add_page()
    chat_pdf.set_font("Arial", size=12)
    
    for message in st.session_state.chat_history:
        chat_pdf.multi_cell(0, 10, message.content)
    
    chat_pdf_output = "chat_log.pdf"
    chat_pdf.output(chat_pdf_output)
    st.download_button("Download Chat as PDF", chat_pdf_output)

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Q&A Assistant", page_icon=":books:", layout="wide")
    st.markdown(css, unsafe_allow_html=True)
    st.title("📄 Chat with Multiple PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Sidebar for PDF Upload and Selection
    with st.sidebar:
        st.header("📂 Upload Your PDFs")
        pdf_docs = st.file_uploader("Select PDFs", accept_multiple_files=True)
        selected_pdf = None

        if pdf_docs:
            st.write(f"Number of PDFs uploaded: {len(pdf_docs)}")
            pdf_names = [pdf.name for pdf in pdf_docs]
            selected_pdf_name = st.selectbox("Select a PDF to analyze:", pdf_names)
            
            for pdf in pdf_docs:
                if pdf.name == selected_pdf_name:
                    selected_pdf = pdf
                    break
            
            if selected_pdf:
                reader = PdfReader(selected_pdf)
                first_page_text = reader.pages[0].extract_text()
                st.text_area("First Page Preview:", first_page_text[:500], height=100)
        
        if st.button("🔍 Analyze") and selected_pdf:
            with st.spinner("Analyzing..."):
                raw_text = get_pdf_text(selected_pdf)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
    # Option to listen for a question
    listen_button = st.button("🎤 Ask with Voice")
    user_question = ""

    if listen_button:
        user_question = listen_for_question()

    # If the user has typed a question
    if not listen_button:
        user_question = st.text_input("💬 Ask about your document:", placeholder="Type your question here...")

    if user_question:
        handle_userinput(user_question)

    # Export options for chat log
    if st.session_state.chat_history:
        st.header("💾 Export Chat")
        export_chat_to_text()
        export_chat_to_pdf()

if __name__ == '__main__':
    main()































