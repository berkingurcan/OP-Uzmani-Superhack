import glob
import os
from uuid import uuid4
import openai
import tiktoken
import pinecone
import json
import time
import chromadb

from chromadb.utils import embedding_functions
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import Pinecone, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

"""
 __    _____    __    ____  ____  _  _  ___ 
(  )  (  _  )  /__\  (  _ \(_  _)( \( )/ __)
 )(__  )(_)(  /(__)\  )(_) )_)(_  )  (( (_-.
(____)(_____)(__)(__)(____/(____)(_)\_)\___/
"""

def md_loader():
    markdown_files = glob.glob(os.path.join("./docs", "*.md"))
    docs = [UnstructuredMarkdownLoader(f, mode = "single").load()[0] for f in markdown_files]
    return docs

loaded_md = md_loader()

def git_loader():
    # TODO: Load github projects
    pass

"""
 ___  ____  __    ____  ____  ____  ____  _  _  ___ 
/ __)(  _ \(  )  (_  _)(_  _)(_  _)(_  _)( \( )/ __)
\__ \ )___/ )(__  _)(_   )(    )(   _)(_  )  (( (_-.
(___/(__)  (____)(____) (__)  (__) (____)(_)\_)\___/
"""

def split_documents():
    # TODO: Splitting
    headers_to_split_on = [
        ("title:", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # TODO: Change chunks to MarkdownHeaderTextSplitter if it is better!
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loaded_md)
    return chunks
   
splitted_documents = split_documents()

"""
 ____  __  __  ____  ____  ____  ____  ____  _  _  ___ 
( ___)(  \/  )(  _ \( ___)(  _ \(  _ \(_  _)( \( )/ __)
 )__)  )    (  ) _ < )__)  )(_) ))(_) )_)(_  )  (( (_-.
(____)(_/\/\_)(____/(____)(____/(____/(____)(_)\_)\___/
"""

def embed():
    # TODO Embedding
    model_name = 'text-embedding-ada-002'

    text = [c.page_content for c in splitted_documents]

    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ['OPENAI_API_KEY']
    ).embed_documents(text)

    length_of_embeddings = (len(embeddings), len(embeddings[0]))

    return embeddings, length_of_embeddings
    
embeddings, length_of_embeddings = embed()
print("LENGTH: ", length_of_embeddings)
# LENGTH:  (197, 1536)

"""
 _  _  ____  ___  ____  _____  ____    ___  ____  _____  ____  ____ 
( \/ )( ___)/ __)(_  _)(  _  )(  _ \  / __)(_  _)(  _  )(  _ \( ___)
 \  /  )__)( (__   )(   )(_)(  )   /  \__ \  )(   )(_)(  )   / )__) 
  \/  (____)\___) (__) (_____)(_)\_)  (___/ (__) (_____)(_)\_)(____)
"""

def vector_store(chunks):

    persist_directory = './db'

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")

    # extract titles from documents
    def extract_title(document):
        lines = document.page_content.split('\n')
        for line in lines:
            if line.startswith('title:'):
                title = line.split('title:')[1].strip()
                return title
        return ""  # Return None if no title is found

    ids = [str(uuid4()) for _ in range(len(splitted_documents))]

    
    collection.add(
        embeddings = embeddings,
        documents = [i.page_content for i in splitted_documents],
        metadatas = [{
            "text": chunks[i].page_content,
            "title": extract_title(chunks[i]),  
        } for i in range(len(chunks))],
        ids = ids
    )

    return collection

collection = vector_store(splitted_documents)
results = collection.query(
    query_texts=["What is op stack"],
    n_results=2
)

print(results)
"""
 ____  ____  ____  ____  ____  ____  _  _  __    __   
(  _ \( ___)(_  _)(  _ \(_  _)( ___)( \/ )/__\  (  )  
 )   / )__)   )(   )   / _)(_  )__)  \  //(__)\  )(__ 
(_)\_)(____) (__) (_)\_)(____)(____)  \/(__)(__)(____)
"""
def retrieval(query):
    # TODO retrival with query

    docs = vector_db.similarity_search(query, k=3)
    return docs[0].page_content


result = retrieval("What is OP Stack?")
print(result)

def print_index_stats():
    index = pinecone.Index('mango')
    print(index.describe_index_stats())


"""
 _____  __  __  ____  ____  __  __  ____ 
(  _  )(  )(  )(_  _)(  _ \(  )(  )(_  _)
 )(_)(  )(__)(   )(   )___/ )(__)(   )(  
(_____)(______) (__) (__)  (______) (__) 
"""

# TODO Chat with your data :D