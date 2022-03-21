# import useful libraries
import os
import numpy as np
import cv2

# function to load our classes names
def read_classes(file):
    """ Read the classes files and extract the classes' names in it"""
    classNames = []
    with open(file, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    return classNames


def object_detection(outputs, input_frame):
    """ This function will allow us to draw bounding boxes on detected objects in frames (images) """
    # first we'll collect the height, width and channel of the input frame (3 if the image is RGB, 1 if it's grayscale)
    height, width, channel = input_frame.shape

    # we'll create category lists to store the layers' output values
    bounding_boxes = []
    class_objects = []
    confidence_probs = []

    # Knowing that there are 3 YOLO layers, we'll browsing them and their outputs using this :
    for result in outputs:
        for values in result:

            # we know that the class probabilities are from the 5th values
            scores = values[5:]
            # get the indice of the max score
            indices_object = np.argmax(scores)
            # store the maximum value of the indice found
            confidence_probability = scores[indices_object]

            # in order to have a proper detection, we'll eliminate the weakest probability detection by imposing a threshold
            if confidence_probability > confidenceThreshold:

                # get the pixel values corresponding to the scaling of the bounding box coordinates to the initial frame
                box_detected = values[0:4] * np.array([width, height, width, height])
                # get the top left corner coordinates by extracting values from box_detected and perform calculations
                x, y, w, h = box_detected
                # we're converting the coordinates to int because OpenCV doesn't allow floats for bounding boxes
                x = int(x - (w/2))
                y = int(y - (h/2))

                # adding the good detected boxe in the bounding boxes list created
                bounding_boxes.append([x, y, w, h])
                # adding the detected objects indices in the class objects list
                class_objects.append(indices_object)
                # adding the max value of the object score (confidence) in the confidences_probs list
                confidence_probs.append(float(confidence_probability))

    return bounding_boxes, class_objects, confidence_probs


def nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold):
    """This function performs non-max suppression on all the bounding boxes detected and keeps the best one"""
    #Using OpenCV DNN non-max supression to get the best bounding box of the detected object (retrieve the indices)
    indices_bbox = cv2.dnn.NMSBoxes(
        bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)
    print('Number of objects detected : ', len(indices_bbox), '\n')

    return indices_bbox


def box_drawing(input_frame, indices, bounding_boxes, class_objects, confidence_probs, classNames, color=(0, 255, 255), thickness=2):
    """ Drawing the detected objects boxes """
    # once we have the indices, we'll extract the values of x,y,w,h of the best bounding boxes and stores it.
    for i in indices:
        # i = i[0]
        final_box = bounding_boxes[i]
    # we'll retrieve the bounding boxes values (coordinates) now and use them to draw our boxes.
        x, y, w, h = final_box[0], final_box[1], final_box[2], final_box[3]
        x, y, w, h = int(x), int(y), int(w), int(h)
        print('Bounding box coordinates in the frame : ', 'x : ',
              x, '|| y : ', y, '|| w : ', w, '|| h :', h, '\n')

        cv2.rectangle(input_frame, (x, y), (x+w, y+h),  color, 2)
        cv2.putText(input_frame, f'{classNames[class_objects[i]].upper()} {int(confidence_probs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6,  color, 2)


def convert_to_blob(input_frame, network, height, width):
    """ This function allow us to convert a frame/image into blob format for OpenCV DNN"""
    blob = cv2.dnn.blobFromImage(input_frame, 1/255, (height, width), [0, 0, 0], 1, crop=False)
    network.setInput(blob)

    # get the YOLO output layers numbers (names), these layers will be useful for the detection part
    # the layer's name : yolo_82, yolo_94, yolo_106
    yoloLayers = network.getLayerNames()
    outputLayers = [yoloLayers[i-1] for i in network.getUnconnectedOutLayers()]

    # Doing forward propagation with OpenCV
    outputs = network.forward(outputLayers)

    return outputs


def load_video(video):
    """ Return the video by inputing the title of the video """
    # get the video path and load the video
    video_path = os.path.join('videos', video)

    return video_path



# test our function read_classes
img_file = './data/coco.names'
classNames = read_classes(img_file)

# load the model config and weights
modelConfig_path = './cfg/yolov3-tiny.cfg'
modelWeights_path = './models/yolov3-tiny.weights'

# read the model cfg and weights with the cv2 DNN module
neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)

# set the preferable Backend to GPU
neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# confidence and non-max suppression threshold for this YoloV3 version
confidenceThreshold = 0.3
nmsThreshold = 0.1

# cap_video = cv2.VideoCapture(load_video('DrivingMiami.mp4')) // --> driving video
cap_video = cv2.VideoCapture(0) # live capture

network = neural_net
# we can decrease the height and width but the minimum is 320x320.
height, width = 320, 320 # this needs to change according to the capture format

# save the video with object detections
frame_width = int(cap_video.get(3))
frame_height = int(cap_video.get(4))
video_frames_save = cv2.VideoWriter('./results/vids/testvid1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

while cap_video.isOpened():
    success, video_frames = cap_video.read()
    # if 'video_frames' is read correctly 'success' is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # using convert_to_blob function :
    outputs = convert_to_blob(video_frames, network, height, width)
    # apply object detection on the video file
    bounding_boxes, class_objects, confidence_probs = object_detection(outputs, video_frames)
    # perform non-max suppression
    indices = nms_bbox(bounding_boxes, confidence_probs,
                       confidenceThreshold, nmsThreshold)
    # draw the boxes
    box_drawing(video_frames, indices, bounding_boxes, class_objects,
                confidence_probs, classNames, color=(0, 255, 255), thickness=2)

    # save the video
    video_frames_save.write(video_frames)

    cv2.imshow('Object Detection in videos', video_frames)

    if cv2.waitKey(1) == ord('q'):
        break

cap_video.release()
cv2.destroyAllWindows()
