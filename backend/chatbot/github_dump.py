import glob
import os
import openai
import tiktoken
import pinecone
import json
import time
import requests
import numpy as np

from uuid import uuid4
from langchain.document_loaders import GitLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.vectorstores import Pinecone
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
loader = GitLoader(
    clone_url="https://github.com/ethereum-optimism/optimism-tutorial",
    repo_path="./repo-docs/",
    branch="main",
)

data = loader.load()

"""
 ___  ____  __    ____  ____  ____  ____  _  _  ___ 
/ __)(  _ \(  )  (_  _)(_  _)(_  _)(_  _)( \( )/ __)
\__ \ )___/ )(__  _)(_   )(    )(   _)(_  )  (( (_-.
(___/(__)  (____)(____) (__)  (__) (____)(_)\_)\___/
"""



"""
 ____  __  __  ____  ____  ____  ____  ____  _  _  ___ 
( ___)(  \/  )(  _ \( ___)(  _ \(  _ \(_  _)( \( )/ __)
 )__)  )    (  ) _ < )__)  )(_) ))(_) )_)(_  )  (( (_-.
(____)(_/\/\_)(____/(____)(____/(____/(____)(_)\_)\___/
"""



"""
 ___  ____  _____  ____  ____ 
/ __)(_  _)(  _  )(  _ \( ___)
\__ \  )(   )(_)(  )   / )__) 
(___/ (__) (_____)(_)\_)(____)
"""