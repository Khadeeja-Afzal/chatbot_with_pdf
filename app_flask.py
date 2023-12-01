from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import bot_template, user_template, css   

app = Flask(__name__)

#Environment VARIABLE SETUP
os.environ["OPENAI_API_KEY"] = "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"
api_key = os.environ["OPENAI_API_KEY"]


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

def handle_user_input(question, conversation):
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
    raw_text = get_pdf_text(uploaded_files)
    text_chunks = get_chunk_text(raw_text)
    vector_store = get_vector_store(text_chunks)
    app.config['conversation'] = get_conversation_chain(vector_store)
    app.config['chat_history'] = None
    return jsonify({'reply': 'PDFs uploaded successfully'})

@app.route('/chat', methods=['POST'])
def chat():
    question = request.json['question']
    bot_messages = handle_user_input(question, app.config['conversation'])
    print(bot_messages)
    print(type(bot_messages))
    return jsonify({'reply': bot_messages})


if __name__ == '__main__':
    load_dotenv()

    # Setup the Flask app
    app.config['conversation'] = None
    app.config['chat_history'] = None
    app.run(debug=True)
