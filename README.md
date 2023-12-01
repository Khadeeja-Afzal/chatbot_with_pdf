# chatbot_with_pdf
PDF Chatbot, using Lang chain and Pinecone, seamlessly extracts and indexes text from PDF documents, enabling users to pose natural language queries for precise question-answering.
PDFbot is a Flask-based chatbot application designed to extract text from PDF documents, create embeddings using OpenAI and Pinecone, and perform similarity searches for answering user queries.

## Key Components:
#Flask Web Application:
   - The main application is built using Flask, a Python web framework, and exposes several endpoints for interacting with the PDFbot functionality.
#PDF Processing:
   - The code utilizes the PyMuPDF library (fitz) to extract text from a PDF document. It opens the PDF file, extracts text from each page, and removes duplicate texts.
#OpenAI and Langchain Integration:
   - OpenAI is used for text embeddings, and Langchain is employed for question-answering capabilities. The OpenAI embeddings model ("text-embedding-ada-002") is initialized using the provided API key.
#Pinecone Vector Store:
   - Pinecone is employed for creating vectors from the extracted texts and performing similarity searches. The code initializes Pinecone with the provided API key and environment settings.
#Web Endpoints:
   - The Flask app defines several endpoints, including:
      - `/`: Home endpoint returning a simple message.
      - `/chatbot`: POST endpoint for receiving user queries, performing similarity searches, and returning answers.
      - `/delete_index`: DELETE endpoint for deleting a specified Pinecone index.
      - `/delete`: POST endpoint for deleting vectors from the Pinecone index based on document IDs.
      - `/fetch_vectors`: POST endpoint for fetching vectors from the Pinecone index based on document IDs.
      - `/create_index`: POST endpoint for creating a new Pinecone index with specified parameters.

#Usage:
Ensure the required Python libraries are installed
Set the necessary environment variables for OpenAI and Pinecone API keys.
Run the Flask application:
Use the provided endpoints to interact with the PDFbot.

#Additional Notes:
The code assumes a specific PDF file path. Modify the `pdf_path` variable if your file is located elsewhere.
Adjust the provided API keys and environment variables accordingly.
For production use, it is recommended to secure sensitive information and follow best practices for deploying Flask applications.



