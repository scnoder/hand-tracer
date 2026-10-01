from tracer import *

options = define_options(python.BaseOptions(model_asset_path="hand_landmarker.task"), 2, 0.8, 0.8, 0.8)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

track_finger(cap, detector, 12)
cap.release()
cv2.destroyAllWindows()
