import torch
import numpy as np
import cv2
from time import time

class Detection:
    def __init__(self, captue_index, model_name):
        self.capture_index = captue_index
        self.model = self.load_model(model_name)
        # self.model = model_name
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        # create new video streaming object to extract video frame by frame to make prediction
        return cv2.VideoCapture(self.capture_index)
    
    def load_model(self, model_name):
        # load yolov5 model from pytorch hub
        # return : Trained pytorch model

        if model_name:
            model = torch.hub.load('ultralytics/yolov5','custom',path=model_name,force_reload=False)
        else:
            model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    # cord return ([x,y,w,h,conf],..,[x,y,w,h,conf])
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # confidencemote than 0.3
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, thickness=2)
        return frame
    
    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            
            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (416,416))
            frame = cv2.flip(frame, 1)
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time-start_time, 2)
            #print(f"Frames per Sec : {fps}")
            cv2.putText(frame, f'FPS: {int(fps)}', (5,25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


detector = Detection(captue_index=0, model_name='yolov5m.pt')
detector()