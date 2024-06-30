
import cv2

yüz_takip = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


video_link = r'C:\Users\user\Desktop\images\video11.mp4'


cap = cv2.VideoCapture(video_link)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    yüz = yüz_takip.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  
    for (x, y, w, h) in yüz:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

   
    cv2.imshow('Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()