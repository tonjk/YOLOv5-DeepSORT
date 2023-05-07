import cv2

# Load video stream from a file or camera
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('Squat1_8_9.avi')
# Create a window to display the video stream
cv2.namedWindow("Object Tracker")
fps = video.get(cv2.CAP_PROP_FPS) 
# Initialize the object's location
object_location = None

while True:
    print('running')
    # Read a frame from the video stream
    ret, frame = video.read()
    # frame = cv2.flip(frame,1)
    # # If there's an error or the video has ended, break the loop
    # if not ret:
    #     break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If the object's location is not yet known, use a Haar cascade classifier to detect it
    if object_location is None:
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(len(objects))
        # If an object is detected, take the first one as the tracked object
        if len(objects) > 0:
            object_location = objects[0]

    # If the object's location is known, draw a rectangle around it and update its location
    # else:
            (x, y, w, h) = object_location
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            area = round(w*h/10000)
            cv2.putText(frame,str(area),(x+int(w/2),y-15), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            
        
        # Update the object's location by searching a small region around the previous location
            search_region = gray[y:y + h, x:x + w]
            objects = cascade.detectMultiScale(search_region, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If an object is found, update its location
        # if len(objects) > 0:
        #     object_location = (x + objects[0][0], y + objects[0][1], objects[0][2], objects[0][3])
        #     print(object_location)
            

        
    # Display the frame with the detected object
    cv2.imshow("Object Tracker", frame)
    
    # Wait for a key press and exit if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    object_location = None
# Release the video stream and close all windows
video.release()
cv2.destroyAllWindows()