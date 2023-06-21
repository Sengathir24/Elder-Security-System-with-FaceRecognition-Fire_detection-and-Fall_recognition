import cv2
import mediapipe as mp 

def distance(a,b):
    return(int((((a[1]-b[1])**2)+((a[2]-b[2])**2))**0.5))
def midpoint(a,b):
    mid=[0]
    mid.append((a[1]+b[1])//2)
    mid.append((a[2]+b[2])//2)
    return(mid)

mpPose = mp.solutions.pose #mp.solutions.pose module. 
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r"sengathir fall.mp4")
#cap = cv2.VideoCapture(0)
count=0
while True:
    success , frame = cap.read()
    if success:
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
            a=lst[11]
            b=lst[12]
            c=lst[23]
            d=lst[24]
            t1 = lst[12][2]
            t2 =lst[11][2]
            
            leg=midpoint(lst[29],lst[30])
            human_height=distance(leg,lst[0])
            d1=distance(lst[11],lst[24])
            d2=distance(lst[12],lst[23])
            
            logic=0
            if t1>= lst[23][2] or t1>= lst[22][2] or t2>= lst[23][2] or t2>= lst[22][2]:
                logic+=1
            if human_height<140:
                logic+=1
            if d1<79 or d2<79 and distance(lst[24],lst[23])>20:
                logic+=1
            if logic>=2:    
                cv2.putText(frame, "Fall Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 8, cv2.LINE_AA)
                cv2.putText(frame, "Fall Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print("frame height:",h,"frame width",w,"d1:",d1,"d2:",d2,"height:",human_height)

            
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