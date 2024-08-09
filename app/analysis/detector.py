import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from super_gradients.training import models

def get_detector(task):
    """
    Returns the appropriate detector function based on the task name.

    Parameters:
    - task: The name of the task, which is used to determine the correct detector.

    Returns:
    - A detector function that corresponds to the task, along with a boolean flag indicating
      if the detector needs to be updated dynamically.
    """
    if "hand movement" in str.lower(task):
        return mp_hand(), False  # Use MediaPipe hand detector for hand movement tasks.
    elif "leg agility" in str.lower(task):
        return yolo_nas_pose(), False  # Use YOLO NAS pose detector for leg agility tasks.
    elif "finger tap" in str.lower(task):
        return mp_hand(), False  # Use MediaPipe hand detector for finger tap tasks.
    elif "toe tapping" in str.lower(task):
        return test_pose(), False  # Use MediaPipe pose detector for toe tapping tasks.

# MediaPipe pose detector using PoseLandmarker from the vision API
def mp_pose():
    """
    Creates and returns a MediaPipe PoseLandmarker object for detecting pose landmarks.

    Returns:
    - A MediaPipe PoseLandmarker object configured for video processing.
    """
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode  # Define running mode as VIDEO.
    base_options = python.BaseOptions(model_asset_path='app/models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.PoseLandmarker.create_from_options(options)

# Simplified MediaPipe pose detector using the basic pose solution
def test_pose():
    """
    Returns a simple MediaPipe Pose object for detecting pose landmarks.

    Returns:
    - A MediaPipe Pose object for general pose detection.
    """
    return mp.solutions.pose.Pose()

# MediaPipe hand detector using HandLandmarker from the vision API
def mp_hand():
    """
    Creates and returns a MediaPipe HandLandmarker object for detecting hand landmarks.

    Returns:
    - A MediaPipe HandLandmarker object configured for video processing, capable of detecting up to 2 hands.
    """
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode  # Define running mode as VIDEO.
    base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  # Configure the detector to detect up to 2 hands.
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.HandLandmarker.create_from_options(options=options)

# YOLO NAS pose detector using the YOLO NAS model from the SuperGradients library
def yolo_nas_pose():
    """
    Loads and returns a YOLO NAS model for pose detection, pretrained on the COCO dataset.

    Returns:
    - A YOLO NAS pose detection model loaded on the appropriate device (CPU, GPU, or MPS).
    """
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")  # Load the YOLO NAS model.
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Check for Apple Silicon support.
    device = 'cuda' if torch.cuda.is_available() else device  # Prefer CUDA if available, otherwise fallback to CPU.
    model.to(device)  # Move the model to the selected device.
    return model
