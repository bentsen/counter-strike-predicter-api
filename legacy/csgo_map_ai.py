from langchain.text_splitter import CharacterTextSplitter
from pydantic import BaseModel

# Import the necessary modules from langchain
from langdetect import detect, DetectorFactory
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define a Pydantic model to handle incoming data types
class Description(BaseModel):
    description: str


# load the document and split it into chunks
loader = CSVLoader(file_path='csgo_map.csv', metadata_columns=['map_name'], source_column='description')
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Initialize the embeddings model
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize Chroma
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
)

# Initialize the Ollama language model
llm = Ollama(model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use five sentences maximum. Keep the answer as concise as possible. 

{context}

Question: {question}

Helpful Answer:
"""

prompt = PromptTemplate.from_template(template)
chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt})


def handle_query(description):
    # Detect language and handle non-English inputs
    if detect(description) != 'en':
        return "Sorry, I currently support only English descriptions."

    # Process the description with the retrieval chain
    response = chain({"query": description})
    return response["result"]