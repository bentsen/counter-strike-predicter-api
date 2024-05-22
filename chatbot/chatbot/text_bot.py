import os
import logging

import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from chatbot.config import RESOURCES_DIR

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_csv_column(file_path):
    df = pd.read_csv(file_path, nrows=0)
    columns = df.columns.tolist()
    source_column = columns[0]
    metadata_columns = columns[1:]
    return source_column, metadata_columns


def load_and_split_documents(file_path, chunk_size=1000, chunk_overlap=100):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        source_column, metadata_columns = get_csv_column(file_path)

        loader = CSVLoader(file_path=file_path, metadata_columns=metadata_columns, source_column=source_column,
                           encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def create_prompt_template():
    template_str = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use around two sentences. 

    {context}

    Question: {question}

    Helpful Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=template_str)


def initialize_chain():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("API key for OpenAI must be set in environment variables.")
        return None

    try:
        embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
        documents = load_and_split_documents(os.path.join(RESOURCES_DIR, 'csv', 'CSGO-Weapons-Data.csv'))

        if not documents:
            logger.error("No documents were loaded.")
            return None

        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings_model)
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key,
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        prompt_template = create_prompt_template()
        chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True,
                                            chain_type_kwargs={"prompt": prompt_template})
        return chain
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


def handle_query(description, chain):
    response = chain({"query": description})
    return response["result"]
