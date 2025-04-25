import cv2

frame_number = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video = cv2.VideoCapture('test_videos/video1.mp4')

if not video.isOpened():
    print("Error: Could not open video.")
else:
    print("Opening video file...")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1
        print(f"Processing frame {frame_number}...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show frame with detected faces
        cv2.imshow('Frame', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

print("Video processing completed!")
