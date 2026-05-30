import cv2
from transformers import pipeline
from PIL import Image

# Load emotion detector (no TensorFlow needed)
emotion_detector = pipeline("image-classification", model="trpakov/vit-face-expression")

# Load the face detection classifier
a = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Allow access to camera for capturing video
b = cv2.VideoCapture(0)
if not b.isOpened():
    print("ERROR: Camera not found!")
else:
    print("Camera opened successfully!")

while True:
    # Reading and detecting face
    c_rec, d_image = b.read()

    if not c_rec:
        break

    # Converting BGR image to grayscale
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    f = a.detectMultiScale(e, 1.3, 6)

    for (x1, y1, w1, h1) in f:  # x axis, y axis, width, height

        # Draw green rectangle around face
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 10)

        # Crop only the face region for emotion analysis
        face_crop = d_image[y1:y1 + h1, x1:x1 + w1]

        try:
            # Convert face crop to PIL image for transformers
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)

            # Analyze emotion
            result = emotion_detector(pil_face)

            # Get dominant emotion: happy, sad, angry, surprise, fear, disgust, neutral
            mood = result[0]['label']

            # Get all emotion scores as percentages
            emotions = {r['label']: r['score'] * 100 for r in result}

        except Exception:
            mood = "Unknown"
            emotions = {}

        # Display dominant mood label above the rectangle
        cv2.putText(
            d_image,
            f"Mood: {mood.upper()}",
            (x1, y1 - 10),              # Position just above the box
            cv2.FONT_HERSHEY_SIMPLEX,   # Font style
            0.9,                        # Font size
            (0, 255, 255),              # Yellow color
            2                           # Thickness
        )

        # Display all emotion scores on the left side of the screen
        y_offset = 30
        cv2.putText(d_image, "Emotions:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for emotion, score in emotions.items():
            y_offset += 25
            bar_length = int(score * 2)  # Scale score to bar length

            # Draw background bar
            cv2.rectangle(d_image, (10, y_offset - 15), (210, y_offset), (50, 50, 50), -1)

            # Draw filled bar based on score
            cv2.rectangle(d_image, (10, y_offset - 15), (10 + bar_length, y_offset), (0, 200, 100), -1)

            # Draw emotion name and percentage
            cv2.putText(d_image, f"{emotion}: {score:.1f}%", (10, y_offset - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow('Mood Detector', d_image)

    # Wait to detect image; press ESC key (key code 27) to exit
    h = cv2.waitKey(40) & 0xFF
    if h == 40:  # ESC key
        break

# Release camera and close all windows
b.release()
cv2.destroyAllWindows()