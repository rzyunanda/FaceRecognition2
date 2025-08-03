# face_app.py (v1.1)
"""
Streamlit UI untuk ABSEN ANTI NITIP

Perubahan v1.1:
• Menambah kolom teks untuk memasukkan **nama** saat memilih mode *Daftarkan Wajah*.
• Nama dikirim sebagai argumen `--name` ke `add_faces.py`, sehingga script tidak lagi meminta input di console.

Cara pakai:
$ streamlit run face_app.py
"""

import os
import subprocess
import sys
import pathlib
import streamlit as st

# ------------- konfigurasi -------------
SCRIPT_REGISTER = "add_faces_dl.py"
SCRIPT_RECOGN   = "test_dl.py"

st.set_page_config(
    page_title="Face Recognition App",
    page_icon="🤳",
    layout="centered",
)

st.title("Face Recognition App")
mode = st.radio("Pilih Mode", ("Daftarkan Wajah", "Recognize Wajah"))

# ---------- input nama jika register ----------
if mode == "Daftarkan Wajah":
    name = st.text_input("Masukkan Nama yang Akan Didaftarkan", "")
else:
    name = None

# ---------- tombol mulai ----------
if st.button("Mulai"):
    # Validasi nama
    if mode == "Daftarkan Wajah" and not name.strip():
        st.warning("Silakan isi nama terlebih dahulu.")
        st.stop()

    script     = SCRIPT_REGISTER if mode == "Daftarkan Wajah" else SCRIPT_RECOGN
    script_path = pathlib.Path(__file__).with_name(script)

    if not script_path.exists():
        st.error(f"File {script} tidak ditemukan di folder yang sama!")
        st.stop()

    # Bangun argumen proses
    cmd = [sys.executable, str(script_path)]
    if mode == "Daftarkan Wajah":
        cmd += ["--name", name.strip()]

    st.info("Menjalankan proses … jendela kamera akan terbuka.")
    try:
        proc = subprocess.Popen(cmd)
        proc.wait()
        if proc.returncode == 0:
            st.success("Proses selesai tanpa error.")
        else:
            st.error(f"Script berakhir dengan kode {proc.returncode}")
    except Exception as exc:
        st.exception(exc)

st.markdown("---")

