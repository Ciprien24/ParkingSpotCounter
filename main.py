import pickle
from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False


MODEL = pickle.load(open("/Users/test1/Desktop/TurtleGame/TurtleRace/PythonProject/ParkingLotDetectorAndCounter/model/model.p", "rb"))


def empty_or_not(spot_bgr):
    """Predict if a parking spot is empty or not."""
    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    """Get bounding boxes for parking spots from connected components."""
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):
    
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots



videoPath = '/Users/test1/Desktop/TurtleGame/TurtleRace/PythonProject/ParkingLotDetectorAndCounter/data/parking_1920_1080_loop.mp4'
maskPath = '/Users/test1/Desktop/TurtleGame/TurtleRace/PythonProject/mask_1920_1080.png'
mask = cv2.imread(maskPath, 0)
cap = cv2.VideoCapture(videoPath)
connectedComponents = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connectedComponents)
print(spots[0])


ret = True
step = 30  
frameNr = 0
spots_status = [None for _ in spots]

while ret:
    ret, frame = cap.read()

    if frame is None:  
        break

    if frameNr % step == 0:
        for spotIndex, spot in enumerate(spots):
            x1, y1, w, h = spot  
            spotCrop = frame[y1:y1 + h, x1:x1 + w]
            if spotCrop.size == 0:
                continue
            spot_status = empty_or_not(spotCrop)
            spots_status[spotIndex] = spot_status
    for spotIndex, spot in enumerate(spots):
        x1, y1, w, h = spot
        spot_status = spots_status[spotIndex]

        if spot_status == EMPTY:
            color = (0, 255, 0)  
        else:
            color = (0, 0, 255)  

    
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    cv2.putText(frame,'Available Parking Spots: {}/ {}'.format(str(sum(spots_status)),str(len(spots_status))),(100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('frame', frame)
    frameNr += 1 
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
