from django.shortcuts import render  # Import render function to render templates
from rest_framework.decorators import api_view  # Import api_view decorator for API views
from rest_framework.response import Response  # Import Response class to return API responses
import cv2  # Import OpenCV for video processing
from django.core.files.storage import FileSystemStorage  # Import FileSystemStorage to handle file uploads
import os  # Import os module for interacting with the operating system
import uuid  # Import uuid to generate unique identifiers
from app.analysis.YOLOTracker import YOLOTracker  # Import YOLOTracker class for video analysis
import time  # Import time module to track processing time
import json  # Import json module to handle JSON data
from app.leg_raise_2 import final_analysis, updatePeaksAndValleys, updateLandMarks  # Import functions for leg raise analysis
import traceback  # Import traceback for error handling and debugging

# View function to render the home page
def home(req):
    return render(req, "index.html")
    # return HttpResponse("<h1>Hello world!</h1>")

# Function to analyze video metadata (duration, frames, fps)
def analyse_video(path=None):
    if path is None:
        return 0, 0

    try:
        # Open the video file using OpenCV
        data = cv2.VideoCapture(path)
    except Exception as e:
        print(f"Error in initialising cv2 with the video path : {e}")
        return 0, 0, 0

    # Count the number of frames in the video
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get the frames per second (fps) of the video
    fps = data.get(cv2.CAP_PROP_FPS)

    if int(frames) == 0 or int(fps) == 0:
        return 0, 0, 0

    # Calculate the duration of the video in seconds
    seconds = round(frames / fps)

    return seconds, frames, fps

# Function to analyze video frames using YOLO tracker
def analyse_video_frames(path=None):
    if path is None:
        return {}

    try:
        print("analysis started")
        start_time = time.time()  # Record the start time of the analysis
        # Perform the analysis using YOLOTracker
        ouputDict = YOLOTracker(path, 'yolov8n.pt', '')
        print("Analysis Done")
        print("--- %s seconds ---" % (time.time() - start_time))  # Print the duration of the analysis

        return ouputDict
    except Exception as e:
        print(f"Error in processing video : {e}")
        return {'error': str(e)}

# Function to update plot data based on JSON input
def update_plot_data(json_data):
    try:
        print("updating plot started")
        start_time = time.time()  # Record the start time of the update
        # Update peaks and valleys in the plot data
        outputDict = updatePeaksAndValleys(json_data)
        print("updating the plot is Done")
        print("--- %s seconds ---" % (time.time() - start_time))  # Print the duration of the update

        return outputDict
    except Exception as e:
        print(f"Error in processing update_plot_data : {e}")
        return {'error': str(e)}

# Function to analyze leg raise video data
def leg_analyse_video(json_data, path=None):
    if path is None:
        return {}

    try:
        print("analysis started")
        start_time = time.time()  # Record the start time of the analysis
        # Perform the final analysis of the leg raise video
        outputDict = final_analysis(json_data, path)
        print("Analysis Done")
        print("--- %s seconds ---" % (time.time() - start_time))  # Print the duration of the analysis

        return outputDict
    except Exception as e:
        print(f"Error in processing video : {e}")
        traceback.print_exc()  # Print the stack trace of the exception
        return {'error': str(e)}

# Function to handle video file upload and trigger analysis
def handle_upload(request):
    if len(request.FILES) == 0:
        raise Exception("No files are uploaded")

    if 'video' not in request.FILES:
        raise Exception("'video' field missing in form-data")

    video = request.FILES['video']
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Generate a unique filename for the uploaded video
    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    folder_path = os.path.join(APP_ROOT, 'uploads')

    file_path = os.path.join(folder_path, file_name)
    # Save the uploaded video file to the specified folder
    FileSystemStorage(folder_path).save(file_name, video)
    print("video saved")

    # Analyze the uploaded video
    val = analyse_video_frames(file_path)
    os.remove(file_path)  # Remove the video file after analysis

    return val

# Function to handle video file upload with additional JSON data and trigger leg raise analysis
def handle_upload2(request):
    if len(request.FILES) == 0:
        raise Exception("No files are uploaded")

    if 'video' not in request.FILES:
        raise Exception("'video' field missing in form-data")

    video = request.FILES['video']
    try:
        # Parse the JSON data from the request
        json_data = json.loads(request.POST['json_data'])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON data")

    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Generate a unique filename for the uploaded video
    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    folder_path = os.path.join(APP_ROOT, 'uploads')

    file_path = os.path.join(folder_path, file_name)
    # Save the uploaded video file to the specified folder
    FileSystemStorage(folder_path).save(file_name, video)
    print("video saved")

    # Analyze the leg raise video with the provided JSON data
    val = leg_analyse_video(json_data, file_path)
    os.remove(file_path)  # Remove the video file after analysis

    return val

# API view to handle video data analysis request
@api_view(['POST'])
def get_video_data(request):
    if request.method == 'POST':
        output = handle_upload(request)  # Handle the file upload and analysis

        return Response(output)  # Return the analysis result as a response

# API view to handle leg raise task analysis request
@api_view(['POST'])
def leg_raise_task(request):
    if request.method == 'POST':
        output = handle_upload2(request)  # Handle the file upload with JSON data and analysis

        return Response(output)  # Return the analysis result as a response

# API view to update plot data based on JSON input
@api_view(['POST'])
def updatePlotData(request):
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            json_data = json.loads(request.POST['json_data'])
        except json.JSONDecodeError:
            raise Exception("Invalid JSON data")

        output = update_plot_data(json_data)  # Update the plot data

        return Response(output)  # Return the updated data as a response

# API view to update landmarks based on JSON input
@api_view(['POST'])
def update_landmarks(request):
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            json_data = json.loads(request.POST['json_data'])
        except json.JSONDecodeError:
            raise Exception("Invalid JSON data")

        output = updateLandMarks(json_data)  # Update the landmarks

        return Response(output)  # Return the updated data as a response
