import glob
import os
import openai
import tiktoken
import pinecone
import json
import time

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import Pinecone


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

    length_of_embedding = len(embeddings[0])

    return embeddings, length_of_embedding
    
embeddings, length_of_embedding = embed()

"""
 _  _  ____  ___  ____  _____  ____    ___  ____  _____  ____  ____ 
( \/ )( ___)/ __)(_  _)(  _  )(  _ \  / __)(_  _)(  _  )(  _ \( ___)
 \  /  )__)( (__   )(   )(_)(  )   /  \__ \  )(   )(_)(  )   / )__) 
  \/  (____)\___) (__) (_____)(_)\_)  (___/ (__) (_____)(_)\_)(____)
"""

def vector_store():
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
    # Set Pinecone environment. Find next to API key in console
    env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"

    pinecone.init(api_key=api_key, environment=env)

    index_name = 'mango'
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=length_of_embedding  # 1536 dim of text-embedding-ada-002
    )

    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pinecone.Index(index_name)
    print(index.describe_index_stats())

    model_name = 'text-embedding-ada-002'

    # TODO: Change embedding to MarkdownHeaderTextSplitter if it is better!

    vector_store = Pinecone(index, embeddings.embed_query, "text")
    return vector_store

vector_stored = vector_store()
print(vector_stored)

"""
 ____  ____  ____  ____  ____  ____  _  _  __    __   
(  _ \( ___)(_  _)(  _ \(_  _)( ___)( \/ )/__\  (  )  
 )   / )__)   )(   )   / _)(_  )__)  \  //(__)\  )(__ 
(_)\_)(____) (__) (_)\_)(____)(____)  \/(__)(__)(____)
"""
def retrieval(query):
    # TODO retrival with query
    openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
    embed_model = "text-embedding-ada-002"

    pass



"""
 _____  __  __  ____  ____  __  __  ____ 
(  _  )(  )(  )(_  _)(  _ \(  )(  )(_  _)
 )(_)(  )(__)(   )(   )___/ )(__)(   )(  
(_____)(______) (__) (__)  (______) (__) 
"""

# TODO Chat with your data :D