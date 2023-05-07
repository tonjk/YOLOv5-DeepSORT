# """
# Face Tracking

# pip uninstall opencv-python
# pip uninstall opencv-contrib-python
# pip install opencv-contrib-python==4.2.0.34 , ONLY
# """

import cv2, sys
import torch

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
        return frame, cord
    
    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()

        
        ht = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        wt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        centroid = [wt//2, ht//2]
        center = [-1, -1]
        
        tracker = cv2.TrackerMedianFlow_create() # cv2 old version
        # tracker = cv2.legacy_TrackerMedianFlow() # cv2 version 4.5.1


        buffer = 100 # ภ้าอยู่ในช่วง +,- ค่า buffer ก็จะยังไม่ทำอะไร ก็จะทำให้กล้องไม่ move บ่อยๆ
        area_buffer = 150*150
        if_loss_contract_in = 10 # sec


        onTracking = False

        speed = 50 # 25, 75, None
        zoom = 100 # 100
        cont = 5

        while True:
            
            ret, frame = cap.read()
            assert ret

            # frame = cv2.resize(frame, (416,416))
            frame = cv2.flip(frame, 1)
            
            if not onTracking:
                start_time = time()
                results = self.score_frame(frame)
                frame, cord = self.plot_boxes(results, frame)
                end_time = time()
                fps = 1/np.round(end_time-start_time, 2)
                # print(cord)
                for (x, y, w , h, c) in cord:
                    if tracker.init(frame, (x, y, w, h)):
                        onTracking = True
            else:
                ok, bbox = tracker.update(frame)
                if ok:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    area = bbox[2] * bbox[3]
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    center = [int(bbox[0] + bbox[2] //2), int(bbox[1] + bbox[3] // 2)]
                    if center[0] - centroid[0] > buffer:
                        print("right")
                        # X.relative_move(pan=cont, tilt=None, zoom=None, speed=speed) # ยิง api right
                    if center[0] - centroid[0] < -buffer:
                        print("left")
                        # X.relative_move(pan=-cont, tilt=None, zoom=None, speed=speed) # ยิง api left
                    if center[1] - centroid[1] > buffer:
                        print("down")
                        # X.relative_move(pan=None, tilt=-cont, zoom=None, speed=speed) # ยิง api downleft
                    if center[1] - centroid[1] < -buffer:
                        print("up")
                        # X.relative_move(pan=None, tilt=cont, zoom=None, speed=speed) # ยิง api up
                else:
                    onTracking = False
                    tracker = cv2.TrackerMedianFlow_create()
                    if if_loss_contract_in == True:
                        # X.go_home_position()
                        print("go_home")

            #print(f"Frames per Sec : {fps}")
            cv2.putText(frame, f'FPS: {int(fps)}', (5,25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
            cv2.imshow('YOLOv5 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


detector = Detection(captue_index=0, model_name='yolov5s.pt')
detector()