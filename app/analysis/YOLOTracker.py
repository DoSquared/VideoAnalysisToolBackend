from ultralytics import YOLO
import cv2
import torch

def YOLOTracker(filePath, modelPath, device='cpu'):
    # Determine if CUDA is available; if so, use GPU for inference, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else device
    # Load the YOLO model with the provided model path
    model = YOLO(modelPath)
    # Open the video file specified by filePath
    cap = cv2.VideoCapture(filePath)

    # Initialize a list to store bounding box information for each frame
    boundingBoxes = []
    # Initialize a counter to keep track of the frame number
    frameNumber = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # Track only objects of a specific class (class 0) using the specified device (CPU or GPU)
            results = model.track(frame, persist=True, classes=[0], verbose=False, device=device)
            data = []

            # Check if there are any detected objects and if the bounding boxes and IDs are available
            if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                # Extract the IDs and bounding boxes of the detected objects
                ind = results[0].boxes.id.cpu().numpy().astype(int)
                box = results[0].boxes.xyxy.cpu().numpy().astype(int)
                # Iterate through each detected object
                for i in range(len(ind)):
                    temp = dict()
                    # Store the object's ID, bounding box coordinates, and dimensions
                    temp['id'] = int(ind[i])
                    temp['x'] = int(box[i][0])
                    temp['y'] = int(box[i][1])
                    temp['width'] = int(box[i][2] - box[i][0])
                    temp['height'] = int(box[i][3] - box[i][1])
                    temp['Subject'] = False  # Initialize a 'Subject' key with a default value of False
                    data.append(temp)

            # Store the results for the current frame, including frame number and detected objects
            frameResults = {'frameNumber': frameNumber, 'data': data}
            boundingBoxes.append(frameResults)
            # print(frameResults)  # Uncomment this line to print the results for each frame (for debugging)

        else:
            # Break the loop if the end of the video is reached
            break

        # Increment the frame number counter
        frameNumber += 1

    # Create an output dictionary containing the video's frame rate and the collected bounding box data
    outputDictionary = dict()
    outputDictionary['fps'] = cap.get(cv2.CAP_PROP_FPS)
    outputDictionary['boundingBoxes'] = boundingBoxes
    # Release the video capture object
    cap.release()
    return outputDictionary
