import cv2
import threading
import playsound
from playsound import playsound

fire_cascade = cv2.CascadeClassifier('D:/Fire_2/Fire-detection-with-AI-and-instant-multi-functional-response-system.-main/fire_detection_cascade_model.xml')

vid = cv2.VideoCapture(0)
runOnce = False

def play_alarm_sound_function():
    try:
        playsound.playsound('D:/Fire_2/Fire-detection-with-AI-and-instant-multi-functional-response-system.-main/Fire_alarm.mp3', True)
        print("Fire alarm end")
    except Exception as e:
        print("Error playing alarm sound:", e)


       
def fire_detection():
    global runOnce

    while True:
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

fire_detection_thread = threading.Thread(target=fire_detection)
fire_detection_thread.start()
