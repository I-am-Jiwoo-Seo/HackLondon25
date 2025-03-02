import cv2
import numpy as np
import pyttsx3  # Text-to-speech library
from ultralytics import YOLO
import easyocr  # OCR library
import winsound  # Beep sound for alerts (Windows)

# ✅ Load YOLOv8 model
model = YOLO("yolov8n.pt")

# ✅ Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# ✅ Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech speed
tts_engine.setProperty('volume', 1.0)  # Max volume

# ✅ Camera Calibration (Estimated)
FOCAL_LENGTH = 600  # Adjust based on your camera
KNOWN_HEIGHTS = {
    "bus": 3.5,  # Meters (approximate real-world height)
    "car": 1.5,  # Meters
    "person": 1.7,  # Meters
}

# ✅ Define Traffic Light Colors
TRAFFIC_LIGHTS = {"red": (0, 0, 255), "green": (0, 255, 0)}

# ✅ Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Perform object detection using YOLOv8
    results = model.predict(source=frame)

    pedestrian_light_status = None  # Track pedestrian light state

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]  # Object class
            
            # ✅ Check if object height is known for distance estimation
            if label in KNOWN_HEIGHTS:
                pixel_height = abs(y2 - y1)  # Height in pixels
                real_height = KNOWN_HEIGHTS[label]  # Real-world height
                
                # ✅ Estimate distance using the pinhole camera model
                distance = (real_height * FOCAL_LENGTH) / pixel_height
                
                # ✅ Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {distance:.2f}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # ✅ Alert if object is approaching 3 meters or less
                if distance <= 3.0:
                    alert_message = f"{label} is approaching 3 metres away"
                    print(alert_message)

                    # ✅ Convert Alert Text to Speech
                    tts_engine.say(alert_message)
                    tts_engine.runAndWait()

                    # ✅ Beep sound (Windows only)
                    winsound.Beep(1000, 500)

                    # ✅ Display alert message on screen
                    cv2.putText(frame, alert_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)

            # ✅ OCR for bus plates and destination
            if label == 'bus':
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                ocr_result = reader.readtext(gray_roi)

                if ocr_result:
                    detected_text = ' '.join([text[1] for text in ocr_result])
                    print(f"Detected Bus Text: {detected_text}")

                    # ✅ Convert Bus Number and Destination to Speech
                    speech_text = f"Bus {detected_text} is approaching"
                    tts_engine.say(speech_text)
                    tts_engine.runAndWait()

                    # ✅ Draw OCR result on the screen
                    cv2.putText(frame, detected_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2)

            # ✅ Recognizing Pedestrian Traffic Lights
            if label == "traffic light":
                roi = frame[y1:y2, x1:x2]
                avg_color = np.mean(roi, axis=(0, 1))  # Get average color in ROI

                # Determine if light is red or green
                if avg_color[2] > avg_color[1]:  # More red than green
                    pedestrian_light_status = "red"
                elif avg_color[1] > avg_color[2]:  # More green than red
                    pedestrian_light_status = "green"

                # ✅ Draw bounding box for traffic light
                cv2.rectangle(frame, (x1, y1), (x2, y2), TRAFFIC_LIGHTS.get(pedestrian_light_status, (255, 255, 255)), 2)

    # ✅ Handle Pedestrian Light Announcements
    if pedestrian_light_status:
        if pedestrian_light_status == "red":
            pedestrian_alert = "Pedestrian light is red. Please wait."
        elif pedestrian_light_status == "green":
            pedestrian_alert = "Pedestrian light is green. You may cross."

        print(pedestrian_alert)
        tts_engine.say(pedestrian_alert)
        tts_engine.runAndWait()

        # Display pedestrian light status
        cv2.putText(frame, pedestrian_alert, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ✅ Display the frame
    cv2.imshow('Object Detection with Distance, OCR & Traffic Light Recognition', frame)

    # ✅ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
