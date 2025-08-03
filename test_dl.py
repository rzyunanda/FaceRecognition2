# test_dl.py
import cv2, pickle, numpy as np, face_recognition, time

DATA_DIR = "data"
encodings = pickle.load(open(f"{DATA_DIR}/embeddings.pkl", "rb"))
names     = pickle.load(open(f"{DATA_DIR}/names.pkl", "rb"))

TOLERANCE = 0.6     # naikkan → longgar, turunkan → ketat
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret: break
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), emb in zip(boxes, encs):
        # --- cari jarak terdekat ---
        dists = np.linalg.norm(encodings - emb, axis=1)
        idx   = np.argmin(dists)
        label = "UNKNOWN"
        if dists[idx] < TOLERANCE:
            label = names[idx]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,128,255), 2)
        cv2.putText(frame, label, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,128,255), 1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
