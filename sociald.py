import cv2 as cv #OpenCV is a great tool for image processing and performing computer vision tasks. It is an open-source library that can be used to perform tasks like face detection, objection tracking, landmark detection, and much more.
from scipy.spatial import distance as dist #It uses numpy underneth. SciPy stands for Scientific Python.It provides more utility functions for optimization, stats and signal processing.
import numpy as np #NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
import argparse #for command-line argument
import imutils #A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges
import os #provides functions for interacting with the operating system

MODEL_PATH = "yolo-coco"

MIN_CONF = 0.3 #minimum confidence
NMS_THRESH = 0.3 #threshold

MIN_DISTANCE = 50 
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
#strip- removes space from beginning and end

print(LABELS)

print(len(LABELS))

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

print(weightsPath)
print(configPath)

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #ArgumentParser() initializes the parser so that you can start to add custom arguments

    
    parser.add_argument('-w', '--weights',default='./yolo-coco/yolov3.weights')
    parser.add_argument('-cfg', '--config',default='./yolo-coco/yolov3.cfg')
    parser.add_argument('-v', '--video-path',default='Test Video.mp4')

    parser.add_argument('-vo', '--video-output-path',default='output_file.avi')

    parser.add_argument('-d', '--display',default=0)
    #default=1, to open the video frame, 0 not open frame
    
    parser.add_argument('-l', '--labels',default='./yolo-coco/coco.names')

    FLAGS, unparsed = parser.parse_known_args()
    #parser.parse_known_args()- returns a namespace and a list of the remaining arguments.
    print(FLAGS)
        
# Get the labels
    LABELS = open(FLAGS.labels).read().strip().split('\n')
    #print(LABELS)


# Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    print(ln)
    #getLayerNames(): Get the name of all layers of the network.
    #getUnconnectedOutLayers(): Get the index of the output layers.
    #CHECK YOLOV3 NETWORK ARCHITECTURE link- https://miro.medium.com/max/2000/1*d4Eg17IVJ0L41e7CTWLLSg.png

if FLAGS.video_path:
# initialize the video stream and pointer to output video file
    print("Reading the test-video")
# open input video if available else webcam stream
    vs = cv.VideoCapture(FLAGS.video_path if FLAGS.video_path else 0)
    #VideoCapture- Open video file or image file sequence or a capturing device or a IP video stream for video capturing.
    writer = None
else:
    print("Can't read the video")

def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize lists of detected bounding boxes, centroids, and confidence
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract teh class ID and confidence(probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes being kept
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results

while True:
    # read the next frame from the input video
    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    # resize the frame and then detect people (only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    # initialize the set of indexes that violate the minimum social distance
    violate = set()
        # ensure there are at least two people detections (required in order to compute the
        # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv.putText(frame, text, (10, frame.shape[0] - 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # check to see if the output frame should be displayed to the screen
    if FLAGS.display > 0:
        # show the output frame
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break

    # if  the video writer has not been  as none
    if writer is None:
        # initialize the video writer
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        print("--> Writing stream to output")
        writer.write(frame)

