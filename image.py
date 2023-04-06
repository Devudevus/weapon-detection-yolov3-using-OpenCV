import cv2
import numpy as np
import sys

# Load YOLOv3
net = cv2.dnn.readNet("D:\VS Code\college project\coco.cfg", "D:\VS Code\college project\coco.weights")

# Load classes
with open("D:\VS Code\college project\labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[layer-1] for layer in net.getUnconnectedOutLayers()]
    
    # Check if output_layers is empty
if not output_layers:
    print("Error: No unconnected layers found in the network.")
    sys.exit()
        
# Set colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load image
img = cv2.imread("D:\VS Code\college project\scissor.jpeg")
height, width, channels = img.shape

# Create blob from image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)

# Pass blob to network
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize variables for boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Iterate over each output layer
for out in outs:
    # Iterate over each detection
    for detection in out:
        # Extract class ID and confidence
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Get center, width, and height of bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Calculate top-left corner of bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            # Append box, confidence, and class ID to lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Iterate over each detected object
for i in range(len(boxes)):
    if i in indexes:
        # Get coordinates and label for bounding box
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        
        # Get color for label
        color = colors[class_ids[i]]
        
        # Draw bounding box and label on image
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
