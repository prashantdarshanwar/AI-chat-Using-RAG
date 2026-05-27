import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 AI PDF Chatbot using RAG")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Upload a PDF and ask questions from it.")

# File Upload
pdf = st.file_uploader("Upload PDF File", type="pdf")

# Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if pdf is not None:

    try:
        # Read PDF
        pdf_reader = PdfReader(pdf)

        text = ""

        # Extract Text
        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

        # Validate Text
        if not text.strip():
            st.error("❌ No readable text found in PDF.")
            st.stop()

        # Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(text)

        # Chunk Validation
        if len(chunks) == 0:
            st.error("❌ No text chunks created.")
            st.stop()

        st.success(f"✅ PDF Processed Successfully! Total Chunks: {len(chunks)}")

        # Convert to Documents
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Embedding Model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create Vector Store
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        # Question Input
        question = st.text_input("💬 Ask a Question from PDF")

        if question:

            with st.spinner("Generating Answer..."):

                # Similarity Search
                relevant_docs = vectorstore.similarity_search(
                    question,
                    k=3
                )

                # Groq LLM
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3
                )

                # QA Chain
                chain = load_qa_chain(
                    llm,
                    chain_type="stuff"
                )

                # Generate Response
                response = chain.run(
                    input_documents=relevant_docs,
                    question=question
                )

                # Save Chat History
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": response
                    }
                )

        # Display Chat History
        if st.session_state.chat_history:

            st.markdown("---")
            st.subheader("🧠 Chat History")

            for chat in reversed(st.session_state.chat_history):

                st.markdown(f"### ❓ Question")
                st.write(chat["question"])

                st.markdown(f"### 🤖 Answer")
                st.write(chat["answer"])

                st.markdown("---")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("📂 Please upload a PDF file to continue.")