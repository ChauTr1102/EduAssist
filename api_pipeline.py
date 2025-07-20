from flask import Flask, request, jsonify
from flask_cors import CORS
from phobert_ollama_text_summarization import VietnameseSummarizationPipeline
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web applications

# Initialize the summarization pipeline
logger.info("Initializing Vietnamese Summarization Pipeline...")
try:
    pipeline = VietnameseSummarizationPipeline()
    logger.info("Pipeline initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

@app.route('/', methods=['GET'])
def home():
    """
    API information endpoint.
    """
    return jsonify({
        "service": "Vietnamese Text Summarization API",
        "version": "1.0.0",
        "status": "running" if pipeline else "error",
        "timestamp": datetime.now().isoformat(),
        "endpoint": {
            "path": "/summarize",
            "method": "POST",
            "description": "Summarize Vietnamese text using the complete pipeline"
        },
        "usage": {
            "request_format": {
                "text": "<Vietnamese text to summarize>",
                "summary_length": "<desired summary length in words, default: 50>"
            },
            "response_format": {
                "success": "boolean",
                "summary": "Vietnamese summary text",
                "processing_time": "time in seconds",
                "timestamp": "ISO timestamp"
            }
        }
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Vietnamese text summarization using the complete pipeline.
    
    Pipeline flow:
    1. Vietnamese text -> English translation
    2. English text -> English summary (using Ollama LLM)
    3. English summary -> Vietnamese summary
    
    Request JSON format:
    {
        "text": "<Vietnamese text to summarize>",
        "summary_length": <desired summary length in words, default: 50>
    }
    
    Response JSON format:
    {
        "success": true,
        "summary": "<Vietnamese summary>",
        "processing_time": <time in seconds>,
        "timestamp": "<ISO timestamp>"
    }
    """
    if not pipeline:
        return jsonify({
            "success": False,
            "error": "Pipeline not initialized",
            "timestamp": datetime.now().isoformat()
        }), 503
    
    try:
        start_time = time.time()
        
        # Parse request JSON
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        vietnamese_text = data.get('text', '').strip()
        summary_length = data.get('summary_length', 50)

        # Validate input
        if not vietnamese_text:
            return jsonify({
                "success": False,
                "error": "Text is required",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        if len(vietnamese_text) < 10:
            return jsonify({
                "success": False,
                "error": "Text too short (minimum 10 characters)",
                "timestamp": datetime.now().isoformat()
            }), 400

        # Process the text through the complete pipeline
        logger.info("Processing Vietnamese text through summarization pipeline...")
        results = pipeline.process(vietnamese_text, summary_length)
        
        processing_time = time.time() - start_time
        
        # Return only the Vietnamese summary
        response = {
            "success": True,
            "summary": results['vietnamese_summary'],
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Summarization completed successfully in {processing_time:.2f} seconds")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Summarization pipeline failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "message": "Use GET / for API information or POST /summarize for text summarization",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("=== Vietnamese Summarization API ===")
    print("Starting server on http://0.0.0.0:5000")
    print("\nAPI Endpoints:")
    print("  GET  /          - API information and usage")
    print("  POST /summarize - Vietnamese text summarization")
    print("\nExample usage:")
    print('  curl -X POST http://localhost:5000/summarize \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"text": "Đây là văn bản tiếng Việt cần tóm tắt...", "summary_length": 50}\'')
    print("\nExample response:")
    print('  {"success": true, "summary": "Tóm tắt tiếng Việt...", "processing_time": 2.34}')
    print("\nMake sure Ollama is running: ollama serve")
    print("Make sure model is available: ollama pull llama3.2:3b")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
