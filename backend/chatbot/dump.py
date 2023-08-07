import glob
import os
import openai
import tiktoken
import pinecone
import json
import time

from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
    markdown_files = glob.glob(os.path.join(path_to_dir, "*.md"))
    docs = [UnstructuredMarkdownLoader(f).load()[0] for f in markdown_files]
    chunks = MarkdownHeaderTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    pass

loader = UnstructuredMarkdownLoader("./docs/understand/landscape.md")

md = loader.load()

"""
 ___  ____  __    ____  ____  ____  ____  _  _  ___ 
/ __)(  _ \(  )  (_  _)(_  _)(_  _)(_  _)( \( )/ __)
\__ \ )___/ )(__  _)(_   )(    )(   _)(_  )  (( (_-.
(___/(__)  (____)(____) (__)  (__) (____)(_)\_)\___/
"""

def split_documents():
    # TODO: Splitting
    pass

"""
 ____  __  __  ____  ____  ____  ____  ____  _  _  ___ 
( ___)(  \/  )(  _ \( ___)(  _ \(  _ \(_  _)( \( )/ __)
 )__)  )    (  ) _ < )__)  )(_) ))(_) )_)(_  )  (( (_-.
(____)(_/\/\_)(____/(____)(____/(____/(____)(_)\_)\___/
"""

def embed():
    # TODO Embedding
    pass

"""
 _  _  ____  ___  ____  _____  ____    ___  ____  _____  ____  ____ 
( \/ )( ___)/ __)(_  _)(  _  )(  _ \  / __)(_  _)(  _  )(  _ \( ___)
 \  /  )__)( (__   )(   )(_)(  )   /  \__ \  )(   )(_)(  )   / )__) 
  \/  (____)\___) (__) (_____)(_)\_)  (___/ (__) (_____)(_)\_)(____)
"""

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
    dimension=1536  # 1536 dim of text-embedding-ada-002
)

# wait for index to be initialized
while not pinecone.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pinecone.Index(index_name)
index.describe_index_stats()


model_name = 'text-embedding-ada-002'
embedding = OpenAIEmbeddings(chunk_size=1)

# TODO: Change embedding to MarkdownHeaderTextSplitter if it is better!
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(md)


vector_store = Pinecone.from_documents(chunks, embedding, index_name=index_name)

"""
 ____  ____  ____  ____  ____  ____  _  _  __    __   
(  _ \( ___)(_  _)(  _ \(_  _)( ___)( \/ )/__\  (  )  
 )   / )__)   )(   )   / _)(_  )__)  \  //(__)\  )(__ 
(_)\_)(____) (__) (_)\_)(____)(____)  \/(__)(__)(____)
"""
def retrieval(query):
    # TODO retrival with query
    pass

openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

embed_model = "text-embedding-ada-002"

"""
 _____  __  __  ____  ____  __  __  ____ 
(  _  )(  )(  )(_  _)(  _ \(  )(  )(_  _)
 )(_)(  )(__)(   )(   )___/ )(__)(   )(  
(_____)(______) (__) (__)  (______) (__) 
"""

# TODO Chat with your data :D