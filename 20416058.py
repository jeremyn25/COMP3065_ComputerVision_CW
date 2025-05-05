import cv2
import numpy as np
import os
import threading
import tkinter as tk
from tkinter import ttk

##
# Create a kalman filter
##
def kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    # Measure position
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # Constant velocity model
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # Process small noise
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
    return kf

##
# Predict the position of all Kalman filters
##
def predict_tracks(kalmanFilters, trackHistory, missingCounts):
    activeTracks = set()
    for personId in list(kalmanFilters.keys()):
        kf = kalmanFilters[personId]
        predicted = kf.predict()
        predictedX, predictedY = int(predicted[0].item()), int(predicted[1].item())
        trackHistory[personId]['predictedPos'] = (predictedX, predictedY)
        missingCounts[personId] += 1 
        activeTracks.add(personId)

##
# Delete the position that are missing for too long
##
def delete_tracks(kalmanFilters, trackHistory, missingCounts, missingFrames):
    # Iterate over a static list to avoid dict-size changes mid-loop
    for track, count in list(missingCounts.items()):
        if count >= missingFrames:
            del kalmanFilters[track]
            del trackHistory[track]
            del missingCounts[track]
##
# Create a draw visualisation of the correct positions
##
def track_visualisation(frame, trackHistory, kalmanFilters):
    for personId in list(kalmanFilters.keys()):
            info = trackHistory[personId]
            if not info.get('detected', False):
                continue

            x, y, w, h = info['box']
            cid = info['classId']
            if cid == 0:
                color = (0, 255, 0)
                label = "Person"
                labelColor = (0, 0, 255)
            elif cid == 2:
                color = (255, 0, 0)
                label = "Car"
                labelColor = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_ITALIC, 0.5, labelColor, 2)
            info['detected'] = False

##
# Main function to run the tracking 
# (Read the input video, run YOLOv4 to detect people and cars, implement Kalman filter for tracking)
##
def run_tracking(inputVideo, outputName, stop_event, status_label):
    status_label.config(text="Status: Processing…")
    vidCapture = cv2.VideoCapture(inputVideo)

    # Load YOLOv4 config and weights
    yoloNet = cv2.dnn.readNet('YOLO/yolov4.weights', 'YOLO/yolov4.cfg')
    layerNames = yoloNet.getLayerNames()
    outputIndices = yoloNet.getUnconnectedOutLayers().flatten() - 1
    outputLayer = []
    for i in outputIndices:
        outputLayer.append(layerNames[i])

    # Set up to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = vidCapture.get(cv2.CAP_PROP_FPS)
    frameWidth = int(vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outputVideo = cv2.VideoWriter(outputName, fourcc, fps, (frameWidth, frameHeight))

    # Initialize Kalman filters and tracking variables
    kalmanFilters = {}
    trackId = 0 
    trackHistory = {}  
    missingCounts = {} 
    alpha = 0.2
    missingFrames = 5

    while vidCapture.isOpened():
        if stop_event.is_set():
            status_label.config(text="Status: Canceled")
            break
        ret, frame = vidCapture.read()
        if not ret:
            break

        # YOLO detection
        blobImage = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True)
        yoloNet.setInput(blobImage)
        outs = yoloNet.forward(outputLayer)
        
        # Extract bounding boxes and confidences
        boxes = []
        confidences = []
        classIds = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > 0.5 and classId in (0, 2):  # Detecting persons and vehicles (classId = 0 and 1)
                    centerX, centerY, width, height = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                    x, y = centerX - width // 2, centerY - height // 2
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        # Use NMS to remove overlapping boxes
        nmsBoxes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        if len(nmsBoxes) > 0:
            if hasattr(nmsBoxes, "flatten"):
                index = nmsBoxes.flatten()
            else:
                index = [i[0] for i in nmsBoxes]
        else:
            index = []

        currDetections = []
        for i in index:
            currDetections.append({
                'classId': classIds[i],
                'box': boxes[i],
                'center': (boxes[i][0] + boxes[i][2] // 2, boxes[i][1] + boxes[i][3] // 2),
                'size': boxes[i][2] * boxes[i][3]
            })

        # Predict the position of existing tracks
        predict_tracks(kalmanFilters, trackHistory, missingCounts)

        # Match Kalman filter predictions with current detections
        detectionMatched = set()
        for personId in list(kalmanFilters.keys()):
            if personId not in trackHistory:
                continue
                
            predictedPos = trackHistory[personId]['predictedPos']
            predictedX, predictedY = predictedPos
            
            bestMatch = None
            minDistance = float('inf')
            
            # Find closest detection
            for i, detection in enumerate(currDetections):
                if i in detectionMatched:
                    continue
                    
                centerX, centerY = detection['center']
                distance = np.sqrt((predictedX - centerX)**2 + (predictedY - centerY)**2)

                # Calculate a dynamic threshold based on bounding box size and frame resolution
                dynamicThreshold = max(100, int(np.sqrt(detection['size']) * 0.5)) 

                if distance < dynamicThreshold and distance < minDistance:
                    minDistance = distance
                    bestMatch = detection
                    bestMatchIndex = i
            
            if bestMatch is not None:
                detectionMatched.add(bestMatchIndex)
                missingCounts[personId] = 0  # Reset missing count
                
                centerX, centerY = bestMatch['center']
                
                # Low‐pass smoothing
                if 'smoothedPosition' in trackHistory[personId]:
                    prevX, prevY = trackHistory[personId]['smoothedPosition']
                    smoothX = int(alpha * centerX + (1 - alpha) * prevX)
                    smoothY = int(alpha * centerY + (1 - alpha) * prevY)
                else:
                    smoothX, smoothY = centerX, centerY
                
                # Update Kalman filter with smoothed position
                measurement = np.array([[np.float32(smoothX)], [np.float32(smoothY)]], np.float32)
                kalmanFilters[personId].correct(measurement)
                
                # Update track
                trackHistory[personId]['classId'] = bestMatch['classId']
                trackHistory[personId]['smoothedPosition'] = (smoothX, smoothY)
                trackHistory[personId]['box'] = bestMatch['box']
                trackHistory[personId]['detected'] = True

        # Create new tracks for unmatched detections
        for i, detection in enumerate(currDetections):
            if i not in detectionMatched:
                # Create a new track
                centerX, centerY = detection['center']
                
                kf = kalman_filter()
                
                # Initialize with current position
                kf.statePost = np.array([[np.float32(centerX)], [np.float32(centerY)], [0], [0]], np.float32)
                
                # Assign new track ID
                kalmanFilters[trackId] = kf
                trackHistory[trackId] = {
                    'classId': detection['classId'], 
                    'smoothedPosition': (centerX, centerY),
                    'box': detection['box'],
                    'detected': True
                }
                missingCounts[trackId] = 0
                trackId += 1

        # Delete tracks that have been missing for too long
        delete_tracks(kalmanFilters, trackHistory, missingCounts, missingFrames)

        # Draw the tracks that are still active
        track_visualisation(frame, trackHistory, kalmanFilters)

        # Write the frame to the output video
        outputVideo.write(frame)

    vidCapture.release()
    outputVideo.release()
    cv2.destroyAllWindows()
    if not stop_event.is_set():
        status_label.config(text="Status: Done!")

# Create the main frame
main = tk.Tk()
main.title("Object Tracker")

stopEvent = threading.Event()

# List of available videos in the Videos folder
ttk.Label(main, text="Video:").grid(row=0,column=0,padx=4,pady=4)
videos = []
for f in os.listdir("Videos"):
    if f.lower().endswith(('.mp4', '.avi', '.mov')):
        videos.append(f)
if videos:
    vidVar = tk.StringVar(value=videos[0])
else:
    vidVar = tk.StringVar(value="")
ttk.Combobox(main,textvariable=vidVar,values=videos,width=30).grid(row=0,column=1,padx=4,pady=4)

# For output name
ttk.Label(main, text="Output name:").grid(row=1,column=0,padx=4,pady=4)
outputEntry = ttk.Entry(main,width=30)
outputEntry.grid(row=1,column=1,padx=4,pady=4)

# Status label of video processing
status = ttk.Label(main, text="Status: Ready") 
status.grid(row=2,column=0,columnspan=2,padx=4,pady=4)

# Start and Exit buttons function
def start():
    status.config(text="Status: Processing...")
    stopEvent.clear()
    inputPath = os.path.join("Videos",vidVar.get())
    outputName=outputEntry.get().strip()
    if not outputName.lower().endswith(('.mp4','.avi')): 
        outputName+=".mp4"
    outputPath = os.path.join("Outputs", outputName)
    threading.Thread(target=run_tracking, args=(inputPath,outputPath,stopEvent, status), daemon=True).start()

def cancel():
    stopEvent.set()
    main.destroy()

buttonFrame=ttk.Frame(main); buttonFrame.grid(row=3,column=0,columnspan=2,pady=6)
ttk.Button(buttonFrame,text="Start",  command=start).pack(side="left",  padx=6)
ttk.Button(buttonFrame,text="Exit",   command=cancel).pack(side="right", padx=6)

main.mainloop()