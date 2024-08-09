import math
import numpy as np
import cv2
from app.analysis.util import filter_signal, get_output  # Import utility functions for filtering signals and generating output.
from app.analysis.detector import get_detector  # Import function to get the appropriate detector based on the task.
from app.analysis.task_analysis import get_essential_landmarks, get_signal, get_normalisation_factor, \
    get_display_landmarks  # Import various analysis functions for processing landmarks and signals.
import scipy.signal as signal

def increase_bounding_box(box, video_w, video_h):
    """
    Increase the size of the bounding box by 25%, while ensuring that the new box
    does not exceed the video frame boundaries.

    Parameters:
    - box: Dictionary containing the original bounding box coordinates and dimensions (x, y, width, height).
    - video_w: Width of the video frame.
    - video_h: Height of the video frame.

    Returns:
    - new_box: Dictionary with the new, enlarged bounding box coordinates and dimensions.
    """
    new_box = {}
    # Expand the bounding box by 12.5% in each direction, ensuring it does not go out of bounds.
    new_box['x'] = int(max(0, box['x'] - box['width'] * 0.125))
    new_box['y'] = int(max(0, box['y'] - box['height'] * 0.125))
    new_box['width'] = int(min(video_w - new_box['x'], box['width'] * 1.25))
    new_box['height'] = int(min(video_h - new_box['y'], box['height'] * 1.25))

    return new_box

def analysis(bounding_box, start_time, end_time, input_video, task_name):
    """
    Perform analysis on a video segment to detect and process essential landmarks within a specified bounding box.

    Parameters:
    - bounding_box: Dictionary containing the original bounding box coordinates and dimensions (x, y, width, height).
    - start_time: Start time of the segment in seconds.
    - end_time: End time of the segment in seconds.
    - input_video: Path to the input video file.
    - task_name: Name of the task, used to determine which detector and processing methods to use.

    Returns:
    - Analysis output: A dictionary containing processed data including landmarks, signals, and normalization factors.
    """
    # Open the video file for processing.
    video = cv2.VideoCapture(input_video)

    # Increase the bounding box size by 25% for better coverage.
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bounding_box = increase_bounding_box(bounding_box, video_width, video_height)

    # Calculate frames corresponding to the start and end times.
    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame_idx = math.floor(fps * start_time)
    end_frame_idx = math.floor(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)  # Set the video position to the start frame.
    current_frame_idx = start_frame_idx

    # Initialize lists to store the detected essential and all landmarks for each frame.
    essential_landmarks = []
    all_landmarks = []

    # Get the appropriate detector for the specified task, along with an update flag.
    detector, detector_update = get_detector(task_name)

    # Process each frame within the specified time segment.
    while current_frame_idx < end_frame_idx:
        # Read the current frame from the video.
        status, current_frame = video.read()

        if status is False:
            break  # Stop if no frame is read (end of video segment or error).

        # Update the detector if needed (dynamic detector update).
        if detector_update:
            detector, _ = get_detector(task_name)

        # Detect essential landmarks within the current frame using the specified bounding box and detector.
        landmarks, allLandmarks = get_essential_landmarks(current_frame, current_frame_idx, task_name, bounding_box, detector)

        # If no landmarks are detected, use the landmarks from the previous frame to maintain continuity.
        if not landmarks:
            try:
                essential_landmarks.append(essential_landmarks[-1])
                all_landmarks.append(all_landmarks[-1])
            except:
                # If it's the first frame and no landmarks are detected, append empty lists.
                essential_landmarks.append([])
                all_landmarks.append([])

            current_frame_idx += 1
            continue  # Move to the next frame.

        # Append detected landmarks for the current frame to the lists.
        essential_landmarks.append(landmarks)
        all_landmarks.append(allLandmarks)
        current_frame_idx += 1

    # Filter out landmarks that don't need to be displayed and calculate normalization factor.
    display_landmarks = get_display_landmarks(essential_landmarks, task_name)
    normalization_factor = get_normalisation_factor(essential_landmarks, task_name)
    
    # Generate the final analysis output.
    return get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time, all_landmarks)

def get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time, all_landmarks):
    """
    Generate the analysis output by processing the detected landmarks and corresponding signals.

    Parameters:
    - task_name: Name of the task, used to determine signal processing methods.
    - display_landmarks: List of landmarks that need to be displayed after processing.
    - normalization_factor: Factor used to normalize the signal of interest.
    - fps: Frames per second of the video (used for time calculations).
    - start_time: Start time of the video segment.
    - end_time: End time of the video segment.
    - all_landmarks: List of all detected landmarks for the entire segment.

    Returns:
    - output: A dictionary containing the processed signal, landmarks, and other analysis data.
    """
    # Generate the signal of interest based on the task and normalize it.
    task_signal = get_signal(display_landmarks, task_name)
    signal_of_interest = np.array(task_signal) / normalization_factor

    # Optionally, filter the signal of interest (commented out by default).
    # signal_of_interest = filter_signal(signal_of_interest, cut_off_frequency=7.5)

    # Calculate the duration of the segment.
    duration = end_time - start_time

    # Upsample the signal to 60 FPS, as some analysis (e.g., peak detection) may work better at higher frame rates.
    fps = 60  # Override FPS to 60 for upsampling.
    time_vector = np.linspace(0, duration, int(duration * fps))
    up_sample_signal = signal.resample(signal_of_interest, len(time_vector))

    # Get the final output by processing the upsampled signal.
    output = get_output(up_sample_signal, duration, start_time)

    # Include landmarks and normalization factor in the output dictionary.
    output['landMarks'] = display_landmarks
    output['allLandMarks'] = all_landmarks
    output['normalization_factor'] = normalization_factor

    return output
