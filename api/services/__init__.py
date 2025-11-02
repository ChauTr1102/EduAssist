# from .whisper import FasterWhisper
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
# from langchain.callbacks import AsyncIteratorCallbackHandler
from api.config import *

# __all__ = ["FasterWhisper"]