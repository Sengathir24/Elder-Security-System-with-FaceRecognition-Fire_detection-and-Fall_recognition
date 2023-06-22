import cv2
import playsound
import threading
from playsound import playsound

fire_cascade = cv2.CascadeClassifier('D:/Fire_2/Fire-detection-with-AI-and-instant-multi-functional-response-system.-main/fire_detection_cascade_model.xml')

vid = cv2.VideoCapture(0)
runOnce = False
exit_flag = False

def play_alarm_sound_function():
    try:
        playsound('D:/Fire_2/Fire-detection-with-AI-and-instant-multi-functional-response-system.-main/Fire_alarm.mp3', True)
        print("Fire alarm end")
    except Exception as e:
        print("Error playing alarm sound:", e)



while True:
    if exit_flag:
        break

    Alarm_Status = False
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

    for (x, y, w, h) in fire:
        cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        print("Fire alarm initiated")
        threading.Thread(target=play_alarm_sound_function).start()

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        exit_flag = True

cv2.destroyAllWindows()
