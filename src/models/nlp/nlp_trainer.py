import os
import logging
import pandas as pd

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from config.settings import OPENAI_API_KEY, DATA_DIR
from src.scrapers.wiki_scraper import scrape_wikipedia

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_chain():
    if not OPENAI_API_KEY:
        logger.error("API key for OpenAI must be set in environment variables.")
        return None

    try:
        embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        wikipedia_data = scrape_wikipedia()
        csgo_data = load_csv_data(os.path.join(DATA_DIR, 'csgo_skins.csv'))

        documents = convert_to_documents(wikipedia_data, csgo_data)

        split_documents = load_and_split_documents(documents)

        if not split_documents:
            logger.error("No documents were loaded.")
            return None

        vectordb = Chroma.from_documents(documents=split_documents, embedding=embeddings_model)
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        prompt_template = create_prompt_template()
        retriever = vectordb.as_retriever(search_kwargs={"k": 50})
        chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True,
                                            chain_type_kwargs={"prompt": prompt_template})
        return chain
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Failed to load CSV data: {e}")
        return []


def convert_to_documents(wikipedia_data, csgo_data):
    documents = []

    # Add Wikipedia data as a single document
    if wikipedia_data:
        documents.append(Document(page_content=wikipedia_data, metadata={"source": "Wikipedia"}))

    # Add CS:GO data from CSV as multiple documents
    for item in csgo_data:
        text = f"Skin Name: {item['skin_name']}\nWeapon: {item['weapon_name']}\nPrice Min: {item['price_min']}\nPrice Max: {item['price_max']}\nPrice Min (StatTrak): {item['price_min_stattrak']}\nPrice Max (StatTrak): {item['price_max_stattrak']}\nImage: {item['image']}"
        documents.append(Document(page_content=text, metadata={"source": "CS:GO Skins CSV"}))

    return documents


def load_and_split_documents(documents, chunk_size=3000, chunk_overlap=500):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(split_docs)} chunks.")
    return split_docs


def create_prompt_template():
    template_str = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    Limit your response to a maximum of five sentences, unless you are asked for a list of skins in the game or details 
    about a weapon. When asked for a list, provide **all known items**, even if it exceeds five sentences.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:

    """
    return PromptTemplate(input_variables=["context", "question"], template=template_str)
