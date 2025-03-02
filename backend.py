from flask import Flask, send_file, request, jsonify, Response
from flask_cors import CORS
import os
import cv2
import numpy as np
import pyttsx3  # Text-to-speech library
import easyocr  # OCR library
from gmaps_crawler import getRouteToDest
from ultralytics import YOLO
import winsound  # Beep sound for alerts (Windows)
import requests
import base64
import time
import shutil


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech speed
tts_engine.setProperty('volume', 1.0)  # Max volume

# Camera Calibration (Estimated)
FOCAL_LENGTH = 600  # Adjust based on your camera
KNOWN_HEIGHTS = {
    "bus": 3.5,  # Meters (approximate real-world height)
    "car": 1.5,  # Meters
    "person": 1.7,  # Meters
    "bicycle": 1.0,  # Meters
}

# Define Traffic Light Colors
TRAFFIC_LIGHTS = {"red": (0, 0, 255), "green": (0, 255, 0)}



def generate_speech(text, output_file):
    """
    Converts text to speech using Google Text-to-Speech API and saves it as an MP3 file.

    Parameters:
    - text (str): The text to be converted into speech.
    - output_file (str): The name of the output MP3 file.

    Returns:
    - output_file (str): The path to the generated speech file.
    """
    
    API_KEY = "AIzaSyBUl3fxlhMXDshN3SV6MbhRGuhTBpNZ_3c"
    URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"

    # Define the request payload
    data = {
        "input": {"text": text},
        "voice": {"languageCode": "en-US", "ssmlGender": "NEUTRAL"},
        "audioConfig": {"audioEncoding": "MP3"}
    }

    # Make the API request
    response = requests.post(URL, json=data)

    # Process the response
    if response.status_code == 200:
        audio_content = response.json()["audioContent"]  # Base64 string
        audio_bytes = base64.b64decode(audio_content)  # Decode to binary

        # Save the audio as an MP3 file
        with open(output_file, "wb") as audio_file:
            audio_file.write(audio_bytes)
        
        print(f"Audio file saved: {output_file}")
        return output_file
    else:
        print("Error:", response.text)
        raise Exception(response.text)



# Open the webcam
cap = cv2.VideoCapture(0)

# Flag to manage if the alert has been spoken already
spoken_alert = False

# Global variable to track the last announced bus
last_announced_bus = None 

def generate_frames():
    global spoken_alert  # Use the flag to control speech
    global last_announced_bus

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection using YOLOv8
        results = model.predict(source=frame)

        pedestrian_light_status = None  # Track pedestrian light state

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]  # Object class
                
                # Check if object height is known for distance estimation
                if label in KNOWN_HEIGHTS:
                    pixel_height = abs(y2 - y1)  # Height in pixels
                    real_height = KNOWN_HEIGHTS[label]  # Real-world height
                    
                    # Estimate distance using the pinhole camera model
                    distance = (real_height * FOCAL_LENGTH) / pixel_height
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Check if the object is within 3 meters and is a "bicycle"
                if distance <= 9.0 and label == "bicycle" and not spoken_alert:
                    alert_message = f"Bicycle is approaching 3 metres away"
                    print(alert_message)

                    # Convert Alert Text to Speech
                    tts_engine.say(alert_message)
                    spoken_alert = True  # Set flag to true, meaning the alert was spoken

                    # Beep sound (Windows only)
                    
                    winsound.Beep(1000, 500)  # Short beep

                    # Display alert message on screen
                    cv2.putText(frame, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)

                # Check if the detected object is a bus
                if label == 'bus':
                    roi = frame[y1:y2, x1:x2]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    ocr_result = reader.readtext(gray_roi)

                    if ocr_result:
                        detected_text = ' '.join([text[1] for text in ocr_result])
                        print(f"Detected Bus Text: {detected_text}")

                        # Check if this bus was already announced
                        if detected_text != last_announced_bus:
                            last_announced_bus = detected_text  # Update last detected bus
                            speech_text = f"The bus for {detected_text} is coming"
                            output_file = "bus_announcement.mp3"

                            # Generate and save speech
                            generate_speech(speech_text, output_file)

                            # Ensure the file is fully written and released
                            time.sleep(1)

                            # Copy the file to avoid locking issues
                            safe_copy = "bus_announcement_copy.mp3"
                            shutil.copy(output_file, safe_copy)

                            # Play the sound using winsound (Ensure it's fully accessible)
                            if os.path.exists(safe_copy):
                                print(f"Playing {safe_copy}...")
                                winsound.PlaySound(safe_copy, winsound.SND_FILENAME)
                            else:
                                print("Error: Audio file not found!")

                        # Draw OCR result on the screen
                        cv2.putText(frame, detected_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 0, 255), 2)

                # Recognizing Pedestrian Traffic Lights
                if label == "traffic light":
                    roi = frame[y1:y2, x1:x2]
                    
                    # Convert to HSV color space for better color detection
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Define range for red color in HSV
                    lower_red = np.array([0, 120, 70])
                    upper_red = np.array([10, 255, 255])
                    red_mask = cv2.inRange(hsv, lower_red, upper_red)

                    # Define range for green color in HSV
                    lower_green = np.array([36, 25, 25])
                    upper_green = np.array([70, 255, 255])
                    green_mask = cv2.inRange(hsv, lower_green, upper_green)

                    # Check if the red or green mask has any significant pixels
                    if np.sum(red_mask) > 1000:  # Threshold to detect red light
                        pedestrian_light_status = "red"
                    elif np.sum(green_mask) > 1000:  # Threshold to detect green light
                        pedestrian_light_status = "green"


                    # Handle Traffic Light Beeping
                    if pedestrian_light_status == "red":
                        print("Red light detected. Playing long beep.")
                        # Long Beep for Red Light
                        winsound.Beep(500, 1500)  # Frequency, duration (longer beep)
                    elif pedestrian_light_status == "green":
                        print("Green light detected. Playing three fast beeps.")
                        # Three fast beeps for Green Light
                        for _ in range(3):
                            winsound.Beep(1000, 300)  # Frequency, duration (shorter beeps)

                    # Draw bounding box for traffic light
                    cv2.rectangle(frame, (x1, y1), (x2, y2), TRAFFIC_LIGHTS.get(pedestrian_light_status, (255, 255, 255)), 2)

        # Handle Pedestrian Light Announcements
        if pedestrian_light_status:
            if pedestrian_light_status == "red":
                pedestrian_alert = "Pedestrian light is red. Please wait."
            elif pedestrian_light_status == "green":
                pedestrian_alert = "Pedestrian light is green. You may cross."

            print(pedestrian_alert)
            tts_engine.say(pedestrian_alert)

            # Display pedestrian light status
            cv2.putText(frame, pedestrian_alert, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # Yield the frame as a byte stream (for HTTP multipart response)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')



@app.route('/')
def home():
    return "Flask is running. Access /video_feed for video stream or /get-mp3 to get MP3 file."

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get-mp3', methods=['POST'])
def set_destination():
    data = request.json
    location = data.get('location')
    destination = data.get('destination')

    if location and destination:
        # Print location and destination for debugging
        print(f"Received location: {location}")
        print(f"Received destination: {destination}")

        # Get the MP3 file path
        mp3_file_path = getRouteToDest(location.get('lat'), location.get('lon'), destination, "nextsteptts.mp3")

        # Check if the file exists
        if os.path.exists(mp3_file_path):
            return send_file(mp3_file_path, mimetype='audio/mpeg', as_attachment=True)
        else:
            return jsonify({"message": "File not found!"}), 404
    else:
        response = {"message": "Both location and destination are required."}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
