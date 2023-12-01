from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import requests 

app = Flask(__name__)

# Environment Variable Setup
os.environ["OPENAI_API_KEY"] = "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"
api_key = os.environ["OPENAI_API_KEY"]

# Replace with your Pinecone API key and URL
PINECONE_API_KEY = "acc743cc-e9f8-489d-b551-dba8ce4e8494"
PINECONE_API_BASE_URL = "https://api.pinecone.io/v1"
pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')

index_name = 'pdfbot'
PINECONE_INDEX_NAME = pinecone.Index(index_name)

# Dictionary to store PDF identifiers and corresponding vectors
pdf_vector_mapping = {}

PINECONE_API_BASE_URL = "https://api.pinecone.io/v1"

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def search_vectors_in_pinecone(query_vector):
    headers = {
        "Authorization": f"Apikey {PINECONE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"query": {"vector": query_vector}}

    response = requests.post(
        f"{PINECONE_API_BASE_URL}/indexes/{PINECONE_INDEX_NAME}/query",
        json=payload,
        headers=headers,
    )

    return response.json()["data"]

def upload_vectors_to_pinecone(vectors):
    headers = {
        "Authorization": f"Apikey {PINECONE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"vectors": vectors}

    try:
        response = requests.post(
            f"{PINECONE_API_BASE_URL}/indexes/{PINECONE_INDEX_NAME}/vectors",
            json=payload,
            headers=headers,
        )
        response.raise_for_status() 

        return response.json()
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        print("Request Exception:", err)

    return {"error": "Failed to upload vectors to Pinecone"}


def handle_user_input(question, conversation):
    c = conversation
    response = conversation({'question': question})
    chat_history = response['chat_history']

    bot_messages = [message.content for i, message in enumerate(chat_history) if i % 2 != 0]

    return bot_messages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file")
    vectors = []

    for pdf_file in uploaded_files:
        pdf_identifier = pdf_file.filename

        if pdf_identifier not in pdf_vector_mapping:
            try:
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()

                text_chunks = get_chunk_text(text)
                embeddings = OpenAIEmbeddings().encode(text_chunks)
                pdf_vector_mapping[pdf_identifier] = embeddings
                vectors.extend(embeddings)

            except Exception as e:
                print("Error processing PDF file:", e)
    
    # Upload vectors to Pinecone
    upload_vectors_to_pinecone(vectors)

    # Add a return statement
    return jsonify({"status": "PDF uploaded successfully"})

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json['question']
    
    # Initialize conversation_chain if not initialized
    if 'conversation' not in app.config or app.config['conversation'] is None:
        app.config['conversation'] = get_conversation_chain(get_vector_store(["dummy_text"]))
    
    bot_messages = handle_user_input(question, app.config['conversation'])
    return jsonify({'reply': bot_messages})

if __name__ == '__main__':
    load_dotenv()

    # Setup the Flask app
    app.run(debug=True)
