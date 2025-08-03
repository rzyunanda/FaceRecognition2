# add_faces_dl.py  ── ganti script lama
import argparse, os, pickle, cv2, face_recognition, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Nama orang yang didaftarkan")
args = parser.parse_args()
name = args.name.strip()

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam tidak terbuka!")

known_encodings, known_names = [], []
total_target, collected = 80, 0

while collected < total_target:
    ret, frame = cap.read()
    if not ret: continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), emb in zip(boxes, encs):
        collected += 1
        known_encodings.append(emb)
        known_names.append(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, f"{collected}/{total_target}", (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)

    cv2.imshow("Face Registration", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release(); cv2.destroyAllWindows()

# ---- simpan embeddings ----
enc_file  = os.path.join(DATA_DIR, "embeddings.pkl")
names_file = os.path.join(DATA_DIR, "names.pkl")

# muat lama bila ada
if os.path.exists(enc_file):
    known_encodings = list(pickle.load(open(enc_file, "rb"))) + known_encodings
    known_names     = list(pickle.load(open(names_file, "rb"))) + known_names

pickle.dump(np.asarray(known_encodings), open(enc_file, "wb"))
pickle.dump(known_names,             open(names_file, "wb"))

print(f"[INFO] selesai: {collected} embedding tersimpan utk {name}")
