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
 ____  _  _  ____  ____  ____  ____  __ _   ___ 
(  __)( \/ )(  _ \(  __)(    \(    \(  ( \ / __)
 ) _) / \/ \ ) _ ( ) _)  ) D ( ) D (/    /( (_ \
(____)\_)(_/(____/(____)(____/(____/\_)__) \___/
"""

def md_loader():
    markdown_files = glob.glob(os.path.join(path_to_dir, "*.md"))
    docs = [UnstructuredMarkdownLoader(f).load()[0] for f in markdown_files]
    chunks = MarkdownHeaderTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    pass

loader = UnstructuredMarkdownLoader("./docs/understand/landscape.md")

md = loader.load()
print(md)


model_name = 'text-embedding-ada-002'
embedding = OpenAIEmbeddings(chunk_size=1)

# TODO: Change embedding to MarkdownHeaderTextSplitter if it is better!
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

""" 
 ____  __  __ _  ____  ___  __   __ _  ____ 
(  _ \(  )(  ( \(  __)/ __)/  \ (  ( \(  __)
 ) __/ )( /    / ) _)( (__(  O )/    / ) _) 
(__)  (__)\_)__)(____)\___)\__/ \_)__)(____) 
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
print(index.describe_index_stats())

