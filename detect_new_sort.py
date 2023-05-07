import torch
import numpy as np
import cv2
from time import time, strftime
from datetime import datetime
import os

# DeepSORT -> Importing DeepSORT.
from deep_sort.application_util import preprocessing, visualization
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

# load model YOlOv5
model_name = 'yolov5s.pt'   #yolov5m.pt// 
model = torch.hub.load('ultralytics/yolov5','custom',
                       path=model_name,
                       force_reload=False)
# define classes
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
# source
capture_index = 0 # from web cam
cap = cv2.VideoCapture(capture_index)



# DeepSORT -> Initializing tracker.
max_cosine_distance = 0.4
nn_budget = None
# model_filename = './model/mars-small128.pb'
model_filename = "./deep_sort/model/mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

frame_idx = 0
# txt_path = os.path.join('sort_txt_file', datetime.now().strftime("%Y%m%d%H%M%S"))
# read frame by frame
while True:

    ret, frame = cap.read()
    # frame = cv2.resize(frame, (416,416))
    frame = cv2.flip(frame, 1)

    # Inference
    start_time = time()

    results = model([frame])
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0] # shape : (height,width,channel)

    
    frame_idx = frame_idx+1
    for i in range(n):
        row = cord[i]

        # DeepSORT -> Extracting Bounding boxes and its confidence scores.
        bboxes = []
        scores = []
        lbs = []
        # for *boxes, conf, cls in det:
        for (*boxes, conf), lb in zip(cord, labels):
            # x1, y1, x2, y2 = int(boxes[0]*x_shape), int(boxes[1]*y_shape), int(boxes[2]*x_shape), int(boxes[3]*y_shape)
            x1, y1, x2, y2 = int(boxes[0]*x_shape), int(boxes[1]*y_shape), int(boxes[2]*x_shape)-int(boxes[0]*x_shape), int(boxes[3]*y_shape-int(boxes[1]*y_shape))
            box = [x1, y1, x2, y2]
            bboxes.append(box)
            scores.append(conf.item())
            lbs.append(classes[int(lb)])
        
        # DeepSORT -> Getting appearance features of the object.
        features = encoder(frame, bboxes)   # im0 >>> frame
        # DeepSORT -> Storing all the required info in a list.
        detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]

        # DeepSORT -> Predicting Tracks.
        tracker.predict()
        tracker.update(detections)

        # DeepSORT -> Plotting the tracks.
        for track,label,scr in zip(tracker.tracks,lbs,scores):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # DeepSORT -> Changing track bbox to top left, bottom right coordinates.
            bbox = list(track.to_tlbr())
            # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
            txt = f"ID:{str(track.track_id)} {label} ({np.round(scr,2)})"
            (label_width,label_height), baseline = cv2.getTextSize(txt , cv2.FONT_HERSHEY_PLAIN,1,1)
            top_left = tuple(map(int,[int(bbox[0]),int(bbox[1])-(label_height+baseline+5)]))
            top_right = tuple(map(int,[int(bbox[0])+label_width,int(bbox[1])]))
            org = tuple(map(int,[int(bbox[0]),int(bbox[1])-baseline]))

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 1)
            cv2.rectangle(frame, top_left, top_right, (255,0,0), -1)
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

            # DeepSORT -> Saving Track predictions into a text file.
            save_format = '{frame},{id},{label},{x1},{y1},{w},{h},{x},{y},{z},{time}\n'
            # print("txt: ", 'sort_txt_file', '.txt')
            with open(f'sort_file_text\\log_text_{strftime("%d%m%y")}.txt', 'a') as f:
                line = save_format.format(frame=frame_idx, id=track.track_id, label=label, x1=int(bbox[0]), y1=int(bbox[1]), w=int(bbox[2]- bbox[0]), h=int(bbox[3]-bbox[1]), x = -1, y = -1, z = -1, time=strftime("%d%m%y%H%M%S"))
                f.write(line)

    #     # draw the rectangle if confidence greater than 0.3
    #     if row[4] >= 0.3:
    #         x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    #         bgr = (0, 255, 0)
    #         cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
    #         cv2.putText(frame, classes[int(labels[i])], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, thickness=2)
    # end_time = time()
    # fps = 1/np.round(end_time-start_time)     
    # cv2.putText(frame, f'FPS: {int(fps)}', (5,25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

    cv2.imshow('YOLOv5 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()