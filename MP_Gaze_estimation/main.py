import mediapipe as mp
import cv2
import gaze
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=0, help='input video path')
    return parser.parse_args()

args = parse_args()
mp_face_mesh = mp.solutions.face_mesh  
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 
cap = cv2.VideoCapture(args.input)

while cap.isOpened():
    success, image = cap.read()
    if not success:  # no frame input
        print("Ignoring empty camera frame.")
        break
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

    if results.multi_face_landmarks:
        gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation

    cv2.imshow('output window', image)
    if cv2.waitKey(2) & 0xFF == 27:
        break
cap.release()
