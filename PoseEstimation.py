import cv2
import mediapipe as mp 

mpPose = mp.solutions.pose #mp.solutions.pose module. 
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r"Pose Estimation\Video\videoplayback.mp4")
#cap = cv2.VideoCapture(0)
while True:
    success , frame = cap.read()
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #Drawing
    if (results.pose_landmarks):
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        lst = []
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = frame.shape
            cx ,cy = int(lm.x* w) , int(lm.y*h) #pixel value
            lst.append([id,cx,cy])
        if lst[23][2] >=250 or lst[22][2] >=250:
            cv2.putText(frame, "Fall Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, "Fall Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        print(lst[23])
        cv2.circle(frame,(lst[23][1],lst[23][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[27][1],lst[27][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[25][1],lst[25][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[32][1],lst[32][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[28][1],lst[28][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[30][1],lst[30][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[29][1],lst[29][2]),5,(255,0,0),cv2.FILLED)
        cv2.circle(frame,(lst[31][1],lst[31][2]),5,(255,0,0),cv2.FILLED)

    cv2.imshow("Frame",frame)
    cv2.waitKey(10)
