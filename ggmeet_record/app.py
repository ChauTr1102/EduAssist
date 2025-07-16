from flask import Flask, request, jsonify, render_template_string
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import cv2
import pyautogui
import threading
import time
import os
import re
import logging
from datetime import datetime
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class MeetRecorder:
    def __init__(self):
        self.driver = None
        self.recording = False
        self.video_writer = None
        self.recording_thread = None
        self.output_dir = "recordings"
        
        # Create recordings directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_driver(self):
        """Setup Chrome driver with necessary options"""
        chrome_options = Options()
        chrome_options.add_argument("--use-fake-ui-for-media-stream")
        chrome_options.add_argument("--use-fake-device-for-media-stream")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Allow microphone and camera access
        prefs = {
            "profile.default_content_setting_values.media_stream_mic": 1,
            "profile.default_content_setting_values.media_stream_camera": 1,
            "profile.default_content_setting_values.notifications": 1
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return True
    
    def extract_meeting_id(self, url):
        """Extract meeting ID from Google Meet URL"""
        patterns = [
            r'meet\.google\.com/([a-z0-9\-]+)',
            r'meet\.google\.com/lookup/([a-z0-9\-]+)',
            r'meet\.google\.com/.*?/([a-z0-9\-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def join_meeting(self, meeting_url, display_name="Recorder Bot"):
        """Join Google Meet meeting"""
        try:
            if not self.driver:
                self.setup_driver()
            
            logging.info(f"Joining meeting: {meeting_url}")
            self.driver.get(meeting_url)
            
            # Wait for page to load
            time.sleep(5)
            
            # Handle name input if present
            try:
                name_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[placeholder*='name' i]"))
                )
                name_input.clear()
                name_input.send_keys(display_name)
                logging.info(f"Set display name: {display_name}")
            except:
                logging.info("Name input not found or not required")
            
            # Turn off camera and microphone
            try:
                # Try to find and click camera button
                camera_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-tooltip*='camera' i], [aria-label*='camera' i]")
                if "turn off" in camera_btn.get_attribute("aria-label").lower():
                    camera_btn.click()
                    logging.info("Camera turned off")
            except:
                logging.info("Camera button not found")
            
            try:
                # Try to find and click microphone button
                mic_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-tooltip*='microphone' i], [aria-label*='microphone' i]")
                if "turn off" in mic_btn.get_attribute("aria-label").lower():
                    mic_btn.click()
                    logging.info("Microphone turned off")
            except:
                logging.info("Microphone button not found")
            
            # Click join button
            join_selectors = [
                "[data-tooltip*='join' i]",
                "[aria-label*='join' i]",
                "button[jsname]",
                ".NPEfkd",
                ".uArJ5e"
            ]
            
            joined = False
            for selector in join_selectors:
                try:
                    join_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    join_btn.click()
                    logging.info(f"Clicked join button with selector: {selector}")
                    joined = True
                    break
                except:
                    continue
            
            if not joined:
                logging.warning("Could not find join button, trying alternative approach")
                # Try pressing Enter key
                self.driver.find_element(By.TAG_NAME, "body").send_keys("\n")
            
            # Wait for meeting to load
            time.sleep(10)
            
            # Check if we're in the meeting
            current_url = self.driver.current_url
            if "meet.google.com" in current_url:
                logging.info("Successfully joined meeting")
                return True
            else:
                logging.error("Failed to join meeting")
                return False
                
        except Exception as e:
            logging.error(f"Error joining meeting: {str(e)}")
            return False
    
    def start_recording(self, meeting_url, duration_minutes=60):
        """Start recording the meeting"""
        try:
            if not self.join_meeting(meeting_url):
                return False
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            meeting_id = self.extract_meeting_id(meeting_url)
            filename = f"meeting_{meeting_id}_{timestamp}.avi"
            filepath = os.path.join(self.output_dir, filename)
            
            # Get screen dimensions
            screen_width, screen_height = pyautogui.size()
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, 20.0, (screen_width, screen_height))
            
            self.recording = True
            self.recording_thread = threading.Thread(
                target=self._record_screen, 
                args=(duration_minutes * 60, filepath)
            )
            self.recording_thread.start()
            
            logging.info(f"Started recording: {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting recording: {str(e)}")
            return False
    
    def _record_screen(self, duration_seconds, filepath):
        """Record screen for specified duration"""
        start_time = time.time()
        
        while self.recording and (time.time() - start_time) < duration_seconds:
            try:
                # Capture screen
                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # Write frame
                self.video_writer.write(frame)
                
                # Control frame rate
                time.sleep(0.05)  # 20 FPS
                
            except Exception as e:
                logging.error(f"Error during recording: {str(e)}")
                break
        
        self.stop_recording()
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        logging.info("Recording stopped")
    
    def get_recordings(self):
        """Get list of recorded files"""
        recordings = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.avi'):
                filepath = os.path.join(self.output_dir, file)
                size = os.path.getsize(filepath)
                recordings.append({
                    'filename': file,
                    'size': size,
                    'created': os.path.getctime(filepath)
                })
        
        return sorted(recordings, key=lambda x: x['created'], reverse=True)

# Global recorder instance
recorder = MeetRecorder()

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Google Meet Recorder</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
        input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
        .start-btn { background: #4CAF50; color: white; }
        .stop-btn { background: #f44336; color: white; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .recordings { margin-top: 20px; }
        .recording-item { background: white; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Google Meet Recorder</h1>
    
    <div class="container">
        <h3>Start Recording</h3>
        <input type="text" id="meetingUrl" placeholder="Enter Google Meet URL" value="https://meet.google.com/abc-defg-hij">
        <br>
        <input type="number" id="duration" placeholder="Duration (minutes)" value="60" min="1" max="240">
        <br>
        <button class="start-btn" onclick="startRecording()">Start Recording</button>
        <button class="stop-btn" onclick="stopRecording()">Stop Recording</button>
    </div>
    
    <div id="status"></div>
    
    <div class="container">
        <h3>Recordings</h3>
        <button onclick="loadRecordings()">Refresh List</button>
        <div id="recordings"></div>
    </div>

    <script>
        function showStatus(message, isError = false) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<div class="status ${isError ? 'error' : 'success'}">${message}</div>`;
            setTimeout(() => statusDiv.innerHTML = '', 5000);
        }

        async function startRecording() {
            const url = document.getElementById('meetingUrl').value;
            const duration = document.getElementById('duration').value;
            
            if (!url) {
                showStatus('Please enter a meeting URL', true);
                return;
            }
            
            try {
                const response = await fetch('/start_recording', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url, duration: parseInt(duration)})
                });
                
                const result = await response.json();
                showStatus(result.message, !result.success);
            } catch (error) {
                showStatus('Error starting recording: ' + error.message, true);
            }
        }

        async function stopRecording() {
            try {
                const response = await fetch('/stop_recording', {method: 'POST'});
                const result = await response.json();
                showStatus(result.message, !result.success);
            } catch (error) {
                showStatus('Error stopping recording: ' + error.message, true);
            }
        }

        async function loadRecordings() {
            try {
                const response = await fetch('/recordings');
                const recordings = await response.json();
                
                const recordingsDiv = document.getElementById('recordings');
                if (recordings.length === 0) {
                    recordingsDiv.innerHTML = '<p>No recordings found</p>';
                    return;
                }
                
                recordingsDiv.innerHTML = recordings.map(rec => `
                    <div class="recording-item">
                        <strong>${rec.filename}</strong><br>
                        Size: ${(rec.size / 1024 / 1024).toFixed(2)} MB<br>
                        Created: ${new Date(rec.created * 1000).toLocaleString()}
                    </div>
                `).join('');
            } catch (error) {
                showStatus('Error loading recordings: ' + error.message, true);
            }
        }

        // Load recordings on page load
        window.onload = loadRecordings;
    </script>
</body>
</html>
    """)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        data = request.get_json()
        meeting_url = data.get('url')
        duration = data.get('duration', 60)
        
        if not meeting_url:
            return jsonify({'success': False, 'message': 'Meeting URL is required'})
        
        if recorder.recording:
            return jsonify({'success': False, 'message': 'Recording already in progress'})
        
        success = recorder.start_recording(meeting_url, duration)
        
        if success:
            return jsonify({'success': True, 'message': f'Recording started for {duration} minutes'})
        else:
            return jsonify({'success': False, 'message': 'Failed to start recording'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        recorder.stop_recording()
        return jsonify({'success': True, 'message': 'Recording stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/recordings')
def get_recordings():
    try:
        recordings = recorder.get_recordings()
        return jsonify(recordings)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Import numpy for cv2
    import numpy as np
    
    print("Google Meet Recorder starting...")
    print("Make sure you have Chrome browser installed")
    print("Access the web interface at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)