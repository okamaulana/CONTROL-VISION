import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import time

# Definisikan ukuran layar
screen_width, screen_height = pyautogui.size()

# Inisialisasi mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Atur lebar frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Atur tinggi frame
cap.set(cv2.CAP_PROP_FPS, 30)  # Atur frame rate

# Variabel untuk penundaan klik
last_click_time = time.time()
click_interval = 1.0  # Penundaan klik dalam detik
click_threshold = 30  # Ambang batas jarak untuk klik
stable_frame_count = 5  # Jumlah frame yang harus stabil sebelum mengklik

# Buffer untuk smoothing jarak dan posisi
distance_buffer = []
stable_frame_counter = 0

def add_to_buffer(buffer, value, max_len=5):
    buffer.append(value)
    if len(buffer) > max_len:
        buffer.pop(0)
    return np.mean(buffer)

while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()
    
    # Flip frame untuk tampilan yang lebih natural
    frame = cv2.flip(frame, 1)
    
    # Konversi dari BGR ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses frame untuk deteksi tangan
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan pada frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Ambil koordinat jari telunjuk dan ibu jari
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Konversi ke koordinat pixel
            h, w, _ = frame.shape
            index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            
            # Gambar titik pada jari telunjuk dan ibu jari
            cv2.circle(frame, index_finger_tip_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, thumb_tip_coords, 10, (0, 0, 255), -1)
            
            # Hitung jarak antara ibu jari dan jari telunjuk
            distance = np.sqrt((index_finger_tip_coords[0] - thumb_tip_coords[0])**2 +
                               (index_finger_tip_coords[1] - thumb_tip_coords[1])**2)
            
            # Tambahkan jarak ke buffer untuk smoothing
            smoothed_distance = add_to_buffer(distance_buffer, distance)
            
            # Jika jarak antara jari telunjuk dan ibu jari kurang dari threshold
            if smoothed_distance < click_threshold:
                stable_frame_counter += 1
            else:
                stable_frame_counter = 0
            
            # Jika jarak stabil dalam beberapa frame, lakukan klik
            if stable_frame_counter >= stable_frame_count:
                # Periksa waktu untuk klik
                current_time = time.time()
                if current_time - last_click_time >= click_interval:
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.putText(frame, 'Clicked', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Ubah koordinat titik telunjuk ke koordinat layar
            mouse_x = screen_width * index_finger_tip_coords[0] / w
            mouse_y = screen_height * index_finger_tip_coords[1] / h
            
            # Pindahkan kursor
            pyautogui.moveTo(mouse_x, mouse_y)
    
    # Tampilkan frame
    cv2.imshow('Hand Gesture Control', frame)
    
    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan setelah selesai
cap.release()
cv2.destroyAllWindows()
