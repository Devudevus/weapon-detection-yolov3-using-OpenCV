import cv2
import numpy as np
import sys

# Load the YOLOv3 model
net = cv2.dnn.readNetFromDarknet('D:\VS Code\college project\coco.cfg', 'D:\VS Code\college project\coco.weights')
classes = []
with open('D:\VS Code\college project\labels.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[layer-1] for layer in net.getUnconnectedOutLayers()]
    
    # Check if output_layers is empty
if not output_layers:
    print("Error: No unconnected layers found in the network.")
    sys.exit()
        
# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the video frame by frame
    ret, frame = cap.read()

    # Prepare the frame for object detection
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections and draw bounding boxes on the frame
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Store the detection results
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes on the frame
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
