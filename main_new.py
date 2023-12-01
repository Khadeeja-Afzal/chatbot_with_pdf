import os
import fitz
import openai, langchain, pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from flask import Flask, render_template, request, jsonify
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI

# KEYS, MODELS and ENV Related Settings
import os
os.environ["OPENAI_API_KEY"] = "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"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embed_model = "text-embedding-ada-002"


os.environ["PINECONE_API_KEY"] = "acc743cc-e9f8-489d-b551-dba8ce4e8494"
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = "gcp-starter"
     
pinecone.init(      
	api_key='acc743cc-e9f8-489d-b551-dba8ce4e8494',      
	environment='gcp-starter'      
)      
index_name = pinecone.Index('pdfbot')

active_indexes = pinecone.list_indexes()
print(active_indexes)

index_description = pinecone.describe_index("pdfbot")
print(index_description)

index = pinecone.Index("pdfbot")
index_stats_response = index.describe_index_stats()
print(index_stats_response)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Define PDF file path
pdf_path = "C:/Users/NKU/Downloads/Chatbot_with_pdf/2nation_theory.pdf."

# Extract text from PDF file
with open(pdf_path, "rb") as f:
    pdf_content = f.read()

# Use PyMuPDF to extract text from each page
pdf_document = fitz.open("pdf", pdf_content)
book_texts = [page.get_text() for page in pdf_document]
pdf_document.close()

# Remove duplicates from the extracted text
unique_texts = list(set(book_texts))

# Create vectors using Pinecone
#book_docsearch = Pinecone.from_texts([t.page_content for t in book_texts], embeddings, index_name = index_name)
book_docsearch = Pinecone.from_texts(unique_texts, embeddings, index_name="pdfbot")

def fetch_vectors(ids_to_fetch):
    query_response = book_docsearch.search(
        search_type="full_text",
        query="",
        ids=ids_to_fetch
    )

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return 'pdfbot'

@app.route("/chatbot", methods=["POST"])
def chatbot_query():
    # Extract query from request data
    query = request.json["query"]

    # Perform similarity search using Pinecone
    docs = book_docsearch.similarity_search(query)

    # Load QA chain and run it with the retrieved documents
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)

    # Return the answer in JSON format
    return jsonify({"answer": answer})


# /delete_index route
@app.route("/delete_index", methods=["DELETE"])
def delete_index():
    try:
        # Extract the index name from the request data
        index_name = request.json.get("index_name")

        if not index_name:
            return jsonify({"error": "Index name is required"}), 400

        # Check if the index exists
        if index_name not in pinecone.list_indexes():
            return jsonify({"error": f"Index '{index_name}' does not exist"}), 404

        # Delete the specified index in Pinecone
        pinecone.delete_index(index_name)
        return jsonify({"success": f"Index '{index_name}' deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to delete index '{index_name}': {str(e)}"}), 500


@app.route("/delete", methods=["POST"])
def delete_vectors():
    # Extract document IDs to delete from request data
    ids_to_delete = request.json["ids"]

    # Delete vectors in Pinecone index
    delete_response = book_docsearch.delete(ids=ids_to_delete)

    # Return the delete response in JSON format
    return jsonify({"delete_response": delete_response})

@app.route("/update_vector_id", methods=["POST"])
def update_vector_id():
    try:
        # Extract parameters from the request data
        old_vector_id = request.json["old_id"]
        new_vector_id = request.json["new_id"]
        namespace = request.json.get("namespace", "")

        # Verify that the old vector ID is correct
        fetch_response = index.fetch(ids=[old_vector_id], namespace=namespace)
        if not fetch_response or "data" not in fetch_response or not fetch_response["data"]:
            return jsonify({"error": f"Vector with ID '{old_vector_id}' not found"}), 404

        # Verify that the new vector ID is not already in use
        fetch_response = index.fetch(ids=[new_vector_id], namespace=namespace)
        if fetch_response and fetch_response.get("data"):
            return jsonify({"error": f"Vector with ID '{new_vector_id}' already exists"}), 400

        # Fetch the existing vector values and metadata
        fetch_response = index.fetch(ids=[old_vector_id], namespace=namespace)
        existing_vector = fetch_response["data"][0]
        existing_values = existing_vector.get("values", [])
        existing_metadata = existing_vector.get("metadata", {})

        # Update the vector with the new ID
        update_response = index.update(
            id=new_vector_id,
            values=existing_values,
            set_metadata=existing_metadata,
            namespace=namespace
        )

        # Delete the old vector
        delete_response = index.delete(ids=[old_vector_id])

        # Return the update and delete responses in JSON format
        return jsonify({"update_response": update_response, "delete_response": delete_response})

    except Exception as e:
        return jsonify({"error": f"Failed to update vector ID: {str(e)}"}), 500

    
@app.route("/fetch_vectors", methods=["POST"])
def fetch_vectors():
    # Extract document IDs to fetch from request data
    ids_to_fetch = request.json["ids"]

    # Fetch vectors from Pinecone index
    fetch_response = index.fetch(ids=ids_to_fetch)

    # Check if fetch response is successful and data is available
    if fetch_response and fetch_response.get("data"):
        return jsonify({"fetch_response": fetch_response["data"]})
    else:
        return jsonify({"error": "Failed to fetch vectors"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0")