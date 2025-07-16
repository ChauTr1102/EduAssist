from flask import Flask, request, jsonify
from flask_cors import CORS
import pyaudio
import wave
import threading
import time
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_thread = None
        self.frames = []
        self.audio = None
        self.stream = None
        
        # Audio configuration
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
    
    def start_recording(self):
        if self.is_recording:
            return False, "Recording already in progress"
        
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            self.is_recording = True
            self.frames = []
            
            # Start recording in a separate thread
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.start()
            
            return True, "Recording started successfully"
            
        except Exception as e:
            return False, f"Error starting recording: {str(e)}"
    
    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
    
    def stop_recording(self):
        if not self.is_recording:
            return False, "No recording in progress", None
        
        try:
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.audio_thread:
                self.audio_thread.join()
            
            # Stop and close the stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            if self.audio:
                self.audio.terminate()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meeting_recording_{timestamp}.wav"
            filepath = os.path.join('recordings', filename)
            
            # Save the recorded audio as WAV file
            wave_file = wave.open(filepath, 'wb')
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(self.audio.get_sample_size(self.format))
            wave_file.setframerate(self.rate)
            wave_file.writeframes(b''.join(self.frames))
            wave_file.close()
            
            return True, f"Recording saved as {filename}", filepath
            
        except Exception as e:
            return False, f"Error stopping recording: {str(e)}", None
    
    def get_status(self):
        return {
            "is_recording": self.is_recording,
            "duration": len(self.frames) * self.chunk / self.rate if self.frames else 0
        }

# Initialize recorder
recorder = AudioRecorder()

@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    success, message = recorder.start_recording()
    return jsonify({
        "success": success,
        "message": message,
        "is_recording": recorder.is_recording
    })

@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    success, message, filepath = recorder.stop_recording()
    return jsonify({
        "success": success,
        "message": message,
        "filepath": filepath,
        "is_recording": recorder.is_recording
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(recorder.get_status())

@app.route('/api/recordings', methods=['GET'])
def list_recordings():
    recordings = []
    if os.path.exists('recordings'):
        files = os.listdir('recordings')
        wav_files = [f for f in files if f.endswith('.wav')]
        recordings = sorted(wav_files, reverse=True)  # Most recent first
    
    return jsonify({
        "recordings": recordings,
        "count": len(recordings)
    })

@app.route('/')
def index():
    return "Audio Recording API is running! Use the HTML frontend to access the recorder."

@app.route('/favicon.ico')
def favicon():
    return "", 204

if __name__ == '__main__':
    print("Starting Audio Recording Server...")
    print("Make sure you have installed the required packages:")
    print("pip install flask flask-cors pyaudio")
    print("\nAPI Endpoints:")
    print("POST /api/start-recording - Start recording")
    print("POST /api/stop-recording - Stop recording and save WAV file")
    print("GET /api/status - Get current recording status")
    print("GET /api/recordings - List all recordings")
    
    app.run(debug=True, host='0.0.0.0', port=5000)