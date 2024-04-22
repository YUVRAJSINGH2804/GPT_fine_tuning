import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import openai
#from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from flask import Flask, render_template, request, redirect

from langchain.retrievers.merger_retriever import MergerRetriever
from text_to_speech import save
from azure.storage.blob import BlobServiceClient

from flask import Flask, request, jsonify
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os




OPENAI_API_KEY = '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'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

#Flask App
app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def home():
    # Render the home page
    return render_template('home.html')

@app.route('/upload-pdf', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        # Assume PDF is uploaded and saved to a known location or Azure
        # For simplicity, let's assume the PDF gets saved locally
        file = request.files['pdf_file']
        filename = secure_filename(file.filename)
        file_path = os.path.join('path/to/save', filename)
        file.save(file_path)

        # Process PDF to extract text and initialize chat
        text = get_pdf_text([file_path])
        text_chunks = get_text_chunks(text)
        global vectorstore
        vectorstore = get_vectorstore(text_chunks)
        global conversation_chain
        conversation_chain = get_conversation_chain(vectorstore)
        return redirect(url_for('chat_page'))
    else:
        # Render the PDF upload page
        return render_template('upload_pdf.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat_page():
    if request.method == 'POST':
        user_input = request.form['user_question']
        response = conversation_chain.respond(user_input)
        chat_history.append(response)
        return render_template('chat.html', chat_history=chat_history)
    else:
        return render_template('chat.html', chat_history=chat_history)

# More functions like get_pdf_text, get_text_chunks etc. should be defined here.

if __name__ == "__main__":
    app.run(debug=True)
