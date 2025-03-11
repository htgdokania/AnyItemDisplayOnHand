import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import threading
import requests
from io import BytesIO
from PIL import Image
import re
from rembg import remove

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Placeholder for current image
current_image = np.zeros((150, 150, 4), dtype=np.uint8)

def fetch_image(query):
    """Fetches and removes background from an image using Google Image search."""
    global current_image
    print(f"Fetching image for: {query}")

    search_url = f"https://www.google.com/search?tbm=isch&q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        # Use regex to find Google thumbnail images
        image_urls = re.findall(r"\"(https://encrypted-tbn0\.gstatic\.com/images\?[^\"<>]+)\"", response.text)

        if image_urls:
            img_url = image_urls[0]
            img_response = requests.get(img_url)
            img_response.raise_for_status()

            # Load image and remove background
            image = Image.open(BytesIO(img_response.content)).convert("RGBA")
            no_bg_image = remove(image)

            # Convert to OpenCV format (BGRA)
            current_image = cv2.cvtColor(np.array(no_bg_image), cv2.COLOR_RGBA2BGRA)
            print(f"Background removed for '{query}'!")

        else:
            print("No valid image found.")
    
    except Exception as e:
        print(f"Image fetch error: {e}")

def overlay_image(bg, overlay, x, y):
    """Overlay transparent image at (x, y), clamped to frame bounds."""
    h, w = overlay.shape[:2]
    x = max(0, min(bg.shape[1] - w, x))
    y = max(0, min(bg.shape[0] - h, y))

    roi = bg[y:y+h, x:x+w]
    alpha = overlay[:, :, 3:] / 255.0
    roi[:, :, :3] = (1 - alpha) * roi[:, :, :3] + alpha * overlay[:, :, :3]
    bg[y:y+h, x:x+w] = roi
    return bg

def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Listening for object names...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command detected: {command}")

                fetch_image(command)

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            print("Didn't catch that, try again...")
        except Exception as e:
            print(f"Error: {e}")

# Start voice recognition thread
voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
voice_thread.start()

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
        middle_finger_mcp = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        # Palm center
        palm_x = int((wrist.x + middle_finger_mcp.x) / 2 * frame.shape[1])
        palm_y = int((wrist.y + middle_finger_mcp.y) / 2 * frame.shape[0])

        # Hand size estimation
        hand_size = int(np.hypot(wrist.x - middle_finger_mcp.x, wrist.y - middle_finger_mcp.y) * frame.shape[1] * 2)
        hand_size = max(50, min(150, hand_size))

        # Resize and overlay image
        resized_image = cv2.resize(current_image, (hand_size, hand_size))
        frame = overlay_image(frame, resized_image, palm_x - hand_size // 2, palm_y - hand_size // 2)

    cv2.imshow("Hand Tracking with Voice Commands", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
