from norfair import Detection, Tracker, Video, draw_tracked_objects
from yolo import YOLO
import cv2


model = YOLO(weights="bestv8.pt")



video = Video(input_path="demo.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=100)
vidout = cv2.VideoWriter("yolo_out.mp4v", -1, int(video.output_fps), (1280, 720))

#print(model.statistics())


for frame in video:

    norfair_detections = model.predict_frame(frame)

    tracked_objects = tracker.update(detections=norfair_detections)
    draw_tracked_objects(frame, tracked_objects)
    #cv2.imshow("VID", frame)
    frame = cv2.resize(frame, (1280, 720))
    vidout.write(frame)
    #cv2.waitKey(1)

