import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker_full.task
# https://google.github.io/mediapipe/solutions/face_mesh.html
# https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api
# storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

def define_options(base_options, hands, hand_detection_confidence, hand_precense_confidence, tracking_confidence):
    return vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=hands,
                                       min_hand_detection_confidence=hand_detection_confidence,
                                       min_hand_presence_confidence=hand_precense_confidence,
                                       min_tracking_confidence=tracking_confidence)


def track_finger(cap, detector, finger):
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
                    index_tip = hand_landmarks[finger]  # index finger tip
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

