import cv2
import numpy as np
import os
import time

# Create a directory to save images and video
if not os.path.exists('detections'):
    os.makedirs('detections')

# Load YOLOv3 model with COCO dataset classes
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Open a video capture object
video_capture = cv2.VideoCapture('video_5.mp4')

# Define output video parameters
output_filename = 'detections/detection_output.mp4'
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

# Define grid parameters
grid_rows = 8
grid_cols = 10
grid_color = (0, 0, 255)  # Red color (BGR format)
grid_alpha = 0.5  # Transparency level

save_interval = 2  # Save images every 2 seconds
last_save_time = time.time()

while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break
    
    # Get dimensions of the frame
    height, width = frame.shape[:2]
    
    # Create a blob from the frame and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(net.getUnconnectedOutLayersNames())
    
    # Extract bounding box information and confidence scores for person class (class_id = 0)
    boxes = []
    
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            
            if class_id == 0 and scores[class_id] > 0.95:  # Person class and confidence > 0.3
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append((x, y, w, h, scores[class_id]))
    
    # Create an overlay for transparent red boxes and grid lines
    overlay = frame.copy()
    
    # Draw grid lines
    for row in range(grid_rows + 1):
        y = row * (height // grid_rows)
        cv2.line(overlay, (0, y), (width, y), grid_color, 2)
    
    for col in range(grid_cols + 1):
        x = col * (width // grid_cols)
        cv2.line(overlay, (x, 0), (x, height), grid_color, 2)
    
    # Iterate over grid boxes
    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = col * (width // grid_cols)
            y1 = row * (height // grid_rows)
            x2 = (col + 1) * (width // grid_cols)
            y2 = (row + 1) * (height // grid_rows)
            
            # Check if any part of a person is in the grid box
            grid_has_person = any(
                x1 <= px <= x1 + w and y1 <= py <= y1 + h
                for x, y, w, h, _ in boxes
                for px in [x, x + w]
                for py in [y, y + h]
            )
            
            # If a person is detected, draw a transparent red rectangle on the overlay
            if grid_has_person:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), grid_color, -1)  # Filled rectangle
    
    # Combine the overlay with the frame to achieve transparency
    frame_with_overlay = cv2.addWeighted(overlay, grid_alpha, frame, 1 - grid_alpha, 0)
    
    # Save images every 2 seconds
    current_time = time.time()
    if current_time - last_save_time >= save_interval:
        image_filename = f'detections/detection_{int(current_time)}.jpg'
        cv2.imwrite(image_filename, frame_with_overlay)
        last_save_time = current_time
    
    # Write the frame with overlay to the output video
    out.write(frame_with_overlay)
    
    # Display the frame with grid and object detections
    cv2.imshow('Grid and Object Detection', frame_with_overlay)
    
    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, video writer, and close all windows
video_capture.release()
out.release()
cv2.destroyAllWindows()

