import cv2
import mediapipe as mp
import math
import pygame

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Initialize pygame mixer
pygame.mixer.init()
sound = pygame.mixer.Sound('sound.mp3')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

fist_detected = False
sound_playing = False

while True:
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(RGB_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                distance = calculate_distance(thumb_tip, index_tip)
                
                if distance < 0.05:  # This threshold may need adjustment
                    if not fist_detected:
                        print("Pinch detected! Playing sound.")
                        if not sound_playing:
                            sound.play()
                            sound_playing = True
                        fist_detected = True
                else:
                    fist_detected = False
                    sound_playing = False
                    pygame.mixer.stop()
        else:
            fist_detected = False
            sound_playing = False
            pygame.mixer.stop()
        
        cv2.imshow("Hand Tracking with Fist Detection and Sound", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
