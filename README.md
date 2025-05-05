# Person Tracking through Videos

This system is a person tracking system that utilize You Look Only Once (YOLO) version 4 for object detection and the Kalman Filter for predicting object movement. It uses a Python GUI with `tkinter` to select input video and custom video output filename.

### Features

- Person and car detection using YOLOv4 object model
- Kalman filter to predict positions frame to frame for smooth tracking
- Video output with bounding box visualisation
- User-friendly GUI to select input video and custom output filename

### Requirements

- Python 3.13+ (I use 3.13.2)
- OpenCV (`opencv-python`)
- NumPy
- `tkinter` (Usually already with Python)
- YOLOv4 configs and weights files (Available in `YOLO/` folder)
- A `Videos/` folder as an input folder that will be detected by the GUI and contains `.mp4`, `.avi`, or `.mov` files
- An `Outputs/` folder to save the processed video

### How to Run

1. `YOLO/` folder should contains `yolov4.weights` and `yolov4.cfg`
2. `Videos/` folder contains video that want to be process
3. Run the script:
   ```bash
   python3 -u 20416058.py
   ```
4. Use the GUI to:
   - Select a video that want to be process from `Videos/` folder
   - Enter an output filename
   - Start the tracking process
   - Wait until the status is `Done!`
5. Processed video will be stored in the `Outputs/` folder

### Output on the Video

- Green Box = Person
- Blue Box = Car  
- Labels are shown with bounding boxes on each frame

