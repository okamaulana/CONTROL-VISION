import cv2
from deepface import DeepFace

# Menggunakan webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi dan analisis wajah
    try:
        # Menggunakan DeepFace untuk analisis
        result = DeepFace.analyze(frame, actions=['age', 'gender'])

        # Ekstrak data usia dan jenis kelamin
        age = result['age']
        gender = result['gender']

        # Tampilkan data di layar
        label = f"Age: {age}, Gender: {gender}"
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Wajah tidak terdeteksi:", e)

    # Tampilkan frame
    cv2.imshow("Age and Gender Estimation", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
