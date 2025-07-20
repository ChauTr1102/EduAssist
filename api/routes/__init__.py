from fastapi import UploadFile, File, Form, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, EmailStr, validator
from dotenv import load_dotenv
import tempfile
from api.services.whisper import FasterWhisper
load_dotenv()

router = APIRouter()
