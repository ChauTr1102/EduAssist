# Vietnamese Text Summarization UI & Backend

This system provides a complete web interface for Vietnamese text summarization using PhoBERT and Ollama models.

## � Project Structure

```
EduAssist/
├── api_pipeline.py                 # Original API route definitions
├── phobert_ollama_text_summarization.py  # Core pipeline
├── requirements.txt                # Dependencies
└── summarization/                  # UI & Backend folder
    ├── __init__.py                # Makes it a Python package
    ├── backend_server.py          # FastAPI server
    ├── gradio_ui.py              # Gradio web interface
    ├── launcher.py               # Startup script
    └── test_api.py               # API testing suite
```

## 🚀 Quick Start

**Multiple ways to start the system:**

### Option 1: Use PowerShell Script (Recommended for Windows)
```powershell
# Navigate to the project and run the PowerShell launcher
cd "d:\FPTU\Summer2025\DSP391m\EduAssist\summarization"
.\start_system.ps1
```

### Option 2: Use Batch File (Windows)
```cmd
# Navigate to the project and run the batch launcher
cd "d:\FPTU\Summer2025\DSP391m\EduAssist\summarization"
start_system.bat
```

### Option 3: Use Python Launcher (Cross-platform)
```bash
# The launcher can now be run from any directory
python "d:\FPTU\Summer2025\DSP391m\EduAssist\summarization\launcher.py"

# Or navigate to the directory first
cd "d:\FPTU\Summer2025\DSP391m\EduAssist\summarization"
python launcher.py
```

### Option 1: Use the Launcher (Recommended)
```bash
python launcher.py
```

This will:
- Check all requirements
- Start the backend API server at http://localhost:8000
- Start the Gradio UI at http://localhost:7860
- Open the UI in your browser automatically

### Option 2: Start Services Manually

1. **Start the Backend Server:**
```bash
python backend_server.py
```

2. **Start the Gradio UI (in another terminal):**
```bash
python gradio_ui.py
```

**Note**: Make sure you're in the `summarization` folder when running these commands!

## 📁 Files Overview

### Core Files
- **`backend_server.py`** - FastAPI server that hosts the summarization API
- **`gradio_ui.py`** - Web UI built with Gradio for easy text summarization
- **`launcher.py`** - Python script to start both services
- **`start_system.ps1`** - PowerShell launcher (Windows)
- **`start_system.bat`** - Batch file launcher (Windows)
- **`test_api.py`** - API testing suite
- **`api_pipeline.py`** - Original API route definitions (imported by backend)

### Dependencies
- **`phobert_ollama_text_summarization.py`** - Core summarization pipeline
- **`requirements.txt`** - Python package requirements

## 🌐 Endpoints

### Backend API (http://localhost:8000)
- **GET `/`** - Root endpoint with API information
- **GET `/health`** - Health check endpoint
- **GET `/api/v1/`** - API information and usage guide
- **POST `/api/v1/summarize`** - Main summarization endpoint
- **GET `/docs`** - Interactive API documentation (Swagger)
- **GET `/redoc`** - Alternative API documentation

### Gradio UI (http://localhost:7860)
- Interactive web interface for text summarization
- Built-in example texts for testing
- Real-time status updates and processing information

## 📝 API Usage

### Summarization Request
```json
POST /api/v1/summarize
{
    "text": "Vietnamese text to summarize...",
    "summary_length": 50
}
```

### Response
```json
{
    "success": true,
    "summary": "Summarized Vietnamese text...",
    "processing_time": 2.34,
    "timestamp": "2025-07-20T10:30:45.123456"
}
```

## 🎯 Features

### Gradio UI Features
- **Clean Interface**: User-friendly design with clear input/output sections
- **Real-time Status**: Shows connection status and processing updates
- **Example Texts**: Pre-loaded examples for quick testing
- **Processing Info**: Detailed metrics about summarization performance
- **Error Handling**: Clear error messages and troubleshooting hints

### Backend Features
- **RESTful API**: Standard HTTP API with JSON responses
- **CORS Support**: Allows frontend access from different origins
- **Health Checks**: Monitoring endpoints for system status
- **Auto-reload**: Development mode with automatic code reloading
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# UI Configuration  
UI_HOST=0.0.0.0
UI_PORT=7860

# Model Configuration (if needed)
MODEL_CACHE_DIR=./models
```

### Custom Ports
To use different ports, modify the files:

**Backend (`backend_server.py`):**
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change port here
```

**UI (`gradio_ui.py`):**
```python
API_BASE_URL = "http://localhost:8001/api/v1"  # Update API URL
interface.launch(server_port=7861)  # Change UI port here
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Cannot connect to API server"**
   - Make sure the backend server is running at http://localhost:8000
   - Check if the port is already in use
   - Verify firewall settings

2. **"Pipeline not initialized"**
   - The PhoBERT/Ollama models are not loaded properly
   - Check the logs in the backend server terminal
   - Ensure all model dependencies are installed

3. **Import Errors**
   - Run: `pip install -r requirements.txt`
   - Make sure you're in the correct Python environment

4. **Port Already in Use**
   - Change ports in the configuration
   - Or stop other services using those ports

### Debug Mode
Start services with debug logging:

```bash
# Backend with debug
python backend_server.py --log-level debug

# UI with debug
python gradio_ui.py  # Debug is enabled by default
```

## 📊 Performance Tips

1. **Summary Length**: Shorter summaries process faster
2. **Text Length**: Very long texts may take more time
3. **Concurrent Requests**: Backend can handle multiple requests
4. **Model Caching**: First request may be slower due to model loading

## 🔒 Security Notes

- **CORS**: Currently allows all origins (`allow_origins=["*"]`)
- **Production**: Change CORS settings for production deployment
- **API Keys**: Add authentication if deploying publicly
- **Input Validation**: API includes basic input validation

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### API Status
```bash
curl http://localhost:8000/api/v1/
```

## 🚀 Deployment

### Development
```bash
python launcher.py
```

### Production (Docker example)
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000 7860
CMD ["python", "launcher.py"]
```

## 📞 Support

For issues or questions:
1. Check the logs in both backend and UI terminals
2. Verify all dependencies are installed correctly  
3. Ensure Ollama service is running (if using Ollama models)
4. Check the API documentation at http://localhost:8000/docs

---

**Built with ❤️ for the EduAssist Project**
