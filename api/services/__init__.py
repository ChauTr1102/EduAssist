# from .whisper import FasterWhisper
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import AsyncIteratorCallbackHandler
from api.config import *

# __all__ = ["FasterWhisper"]