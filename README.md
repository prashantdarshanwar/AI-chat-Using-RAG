# AI PDF Chatbot using RAG

## Overview
This project is an AI-powered PDF chatbot that allows users to upload PDF documents and ask questions related to the document content using Generative AI and RAG architecture.

## Features
- PDF Upload
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- AI-based Question Answering
- Streamlit UI
- Vector Database Integration

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Gemini API
- HuggingFace Embeddings

## Installation

### Create Virtual Environment

Windows:
python -m venv venv

Activate:
venv\Scripts\activate

Linux/Mac:
python3 -m venv venv

Activate:
source venv/bin/activate

## Install Requirements

pip install -r requirements.txt

## Add API Key

Create .env file and add:

GOOGLE_API_KEY=your_api_key

## Run Project

streamlit run app.py

## Architecture

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Retriever
↓
LLM Response

## Future Improvements
- Multiple PDF Upload
- Chat History
- Authentication
- Cloud Deployment
- Advanced Vector Databases

## Author
Prashant Darshanwar