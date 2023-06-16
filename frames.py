import cv2
import os

video_path = r"D:\frames\video\videoplayback.mp4"
output_directory = r"D:\frames\output"

# Create the output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

webcam = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    success, frame = webcam.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Frame Extractor", frame)

    # Save the frame as an image file
    output_path = os.path.join(output_directory, f"frame_{frame_count}.png")
    cv2.imwrite(output_path, gray)
    
    frame_count += 1

    key = cv2.waitKey(27)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
