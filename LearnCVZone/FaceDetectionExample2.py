import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar cascade face detector (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        break

    # Convert frame to grayscale (Haar cascades work on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Draw bounding boxes and centers
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Draw center point
        center = (x + w // 2, y + h // 2)
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Optional: show a dummy confidence score
        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Image', img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
