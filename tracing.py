import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker_full.task
# https://google.github.io/mediapipe/solutions/face_mesh.html
# https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api
# storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1,
                                       min_hand_detection_confidence=0.8,
                                       min_hand_presence_confidence=0.8,
                                       min_tracking_confidence=0.8)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    try:
        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                index_tip = hand_landmarks[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()


# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Ignoring empty camera frame.")
#         continue
    
#     frame = cv2.resize(frame, (640, 480))
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb_frame = cv2.resize(rgb_frame, (640, 480))  # ensure standard dimensions
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#     results = detector.detect(mp_image)

#     if results.hand_landmarks:
#         for hand_landmarks in results.hand_landmarks:
#             index_tip = hand_landmarks[8]
#             x,y = int(index_tip.x * w), int(index_tip.y * h)
#             cv2.circle(frame, (x,y), 10, (255,0,255), -1)

#     cv2.imshow("Hand Tracking", frame)

#     if cv2.waitKey(5) & 0xFF == ord('q'): # quit on 'q' key press
#         break

# cap.release()
# cv2.destroyAllWindows()