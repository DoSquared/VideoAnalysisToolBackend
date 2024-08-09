import math
import numpy as np
import mediapipe as mp
import cv2

import app.analysis.constants.mp_handlandmarks as MP_HAND_LANDMARKS
import app.analysis.constants.mp_landmarks as MP_LANDMARKS
import app.analysis.constants.yolo_landmarks as YOLO_LANDMARKS

from mediapipe.framework.formats import landmark_pb2

def draw_opt(rgb_image, detection_result):
    """
    Draws pose landmarks on the given RGB image using the detection results.

    Parameters:
    - rgb_image: The original image in RGB format.
    - detection_result: The result from the pose detection model containing landmarks.

    Returns:
    - annotated_image: The image with drawn pose landmarks.
    """
    pose_landmarks_list = detection_result.pose_landmarks[0]  # Extract the first pose landmark from the result.
    annotated_image = np.copy(rgb_image)  # Make a copy of the image to draw on.
    
    # Convert the list of pose landmarks to the appropriate protobuf format.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list
    ])
    
    # Draw the landmarks on the image.
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    )
    
    return annotated_image

def draw_hand(rgb_image, hand_landmarks, bounds=None):
    """
    Draws hand landmarks on the given RGB image.

    Parameters:
    - rgb_image: The original image in RGB format.
    - hand_landmarks: The landmarks detected for the hand.
    - bounds: Optional bounding box for the hand, used for scaling landmarks.

    Returns:
    - annotated_image: The image with drawn hand landmarks.
    """
    annotated_image = np.copy(rgb_image)  # Make a copy of the image to draw on.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    try:
        # Convert the hand landmarks to the appropriate protobuf format.
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
    except:
        # If bounds are provided, normalize the landmarks to the bounding box.
        [x1, y1, x2, y2] = bounds
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark[0] / (x2 - x1), y=landmark[1] / (y2 - y1), z=landmark[2]) 
            for landmark in hand_landmarks
        ])
    
    # Draw the hand landmarks on the image.
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto
    )
    
    return annotated_image

def get_essential_landmarks(current_frame, current_frame_idx, task, bounding_box, detector):
    """
    Determines which landmarks are essential based on the task and extracts them.

    Parameters:
    - current_frame: The current video frame.
    - current_frame_idx: The index of the current frame in the video.
    - task: The task being performed (e.g., hand movement, finger tap).
    - bounding_box: The bounding box for the area of interest in the frame.
    - detector: The model used for detecting the landmarks.

    Returns:
    - A tuple containing the essential landmarks and all landmarks.
    """
    is_left = False
    if "left" in str.lower(task):
        is_left = True

    # Depending on the task, call the appropriate function to get the landmarks.
    if "hand movement" in str.lower(task):
        return get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_landmarks(bounding_box, detector, current_frame, current_frame_idx, is_left)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)

def get_signal(display_landmarks, task):
    """
    Extracts the signal based on the landmarks and task.

    Parameters:
    - display_landmarks: The landmarks that need to be displayed.
    - task: The task being performed.

    Returns:
    - The extracted signal.
    """
    # Depending on the task, call the appropriate function to get the signal.
    if "hand movement" in str.lower(task):
        return get_hand_movement_signal(display_landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_signal(display_landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_signal(display_landmarks)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_signal(display_landmarks)

def get_normalisation_factor(landmarks, task):
    """
    Calculates a normalization factor for the signal based on the landmarks and task.

    Parameters:
    - landmarks: The list of detected landmarks.
    - task: The task being performed.

    Returns:
    - The normalization factor.
    """
    # Depending on the task, call the appropriate function to get the normalization factor.
    if "hand movement" in str.lower(task):
        return get_hand_movement_nf(landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_nf(landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_nf(landmarks)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_nf(landmarks)

def get_display_landmarks(landmarks, task):
    """
    Determines which landmarks should be displayed based on the task.

    Parameters:
    - landmarks: The list of detected landmarks.
    - task: The task being performed.

    Returns:
    - The landmarks that should be displayed.
    """
    # Depending on the task, call the appropriate function to get the display landmarks.
    if "hand movement" in str.lower(task):
        return get_hand_movement_display_landmarks(landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_display_landmarks(landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_display_landmarks(landmarks)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_display_landmarks(landmarks)

def get_leg_agility_landmarks(bounding_box, detector, current_frame, current_frame_idx, is_left):
    """
    Extracts the essential landmarks for the leg agility task.

    Parameters:
    - bounding_box: The bounding box for the area of interest in the frame.
    - detector: The model used for detecting the landmarks.
    - current_frame: The current video frame.
    - current_frame_idx: The index of the current frame in the video.
    - is_left: Boolean indicating whether the task involves the left side.

    Returns:
    - A tuple containing selected landmarks and all landmarks for the leg agility task.
    """
    [x1, y1, x2, y2] = get_boundaries(bounding_box)
    roi = current_frame[y1:y2, x1:x2]  # Extract the region of interest (ROI) based on the bounding box.

    # Convert the ROI to RGB since many models expect input in this format.
    results = detector.predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), conf=0.7)
    landmarks = results.prediction.poses[0]  # Extract the first set of pose landmarks.

    knee_idx = YOLO_LANDMARKS.LEFT_KNEE if is_left else YOLO_LANDMARKS.RIGHT_KNEE

    # Extract relevant landmarks for leg agility.
    left_shoulder = landmarks[YOLO_LANDMARKS.LEFT_SHOULDER]
    right_shoulder = landmarks[YOLO_LANDMARKS.RIGHT_SHOULDER]
    knee_landmark = landmarks[knee_idx]
    left_hip = landmarks[YOLO_LANDMARKS.LEFT_HIP]
    right_hip = landmarks[YOLO_LANDMARKS.RIGHT_HIP]

    # Return the selected landmarks and all landmarks in the YOLO format.
    return [left_shoulder, right_shoulder, knee_landmark, left_hip, right_hip], get_all_landmarks_coord_YOLONAS(landmarks)

def get_leg_agility_signal(landmarks_list):
    """
    Generates a signal for the leg agility task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - signal: The generated signal representing the leg movement.
    """
    signal = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, knee_landmark] = landmarks
        distance = math.dist(knee_landmark[:2], shoulder_midpoint)  # Calculate the distance between knee and shoulders.
        signal.append(distance)
    return signal

def get_leg_agility_nf(landmarks_list):
    """
    Calculates the normalization factor for the leg agility task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The average distance between the shoulders and hips, used as the normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, _, left_hip, right_hip] = landmarks
        shoulder_midpoint = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        hip_midpoint = (np.array(left_hip[:2]) + np.array(right_hip[:2])) / 2
        distance = math.dist(shoulder_midpoint, hip_midpoint)  # Calculate the distance between shoulders and hips.
        values.append(distance)
    return np.mean(values)

def get_leg_agility_display_landmarks(landmarks_list):
    """
    Selects landmarks for display during the leg agility task.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The list of landmarks to be displayed.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, knee_landmark, _, _] = landmarks
        shoulder_midpoint = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        display_landmarks.append([shoulder_midpoint, knee_landmark])  # Add shoulder midpoint and knee landmark to display.
    return display_landmarks

def get_toe_tapping_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left):
    """
    Extracts the essential landmarks for the toe tapping task.

    Parameters:
    - bounding_box: The bounding box for the area of interest in the frame.
    - detector: The model used for detecting the landmarks.
    - current_frame_idx: The index of the current frame in the video.
    - current_frame: The current video frame.
    - is_left: Boolean indicating whether the task involves the left side.

    Returns:
    - A tuple containing selected landmarks and all landmarks for the toe tapping task.
    """
    [x1, y1, x2, y2] = get_boundaries(bounding_box)

    frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    # Process the region of interest to detect pose landmarks.
    landmarks = detector.process(frame[y1:y2, x1:x2, :]).pose_landmarks.landmark

    knee_idx = MP_LANDMARKS.LEFT_KNEE if is_left else MP_LANDMARKS.RIGHT_KNEE
    toe_idx = MP_LANDMARKS.LEFT_FOOT_INDEX if is_left else MP_LANDMARKS.RIGHT_FOOT_INDEX

    # Calculate the midpoint between the shoulders.
    left_shoulder = landmarks[MP_LANDMARKS.LEFT_SHOULDER]
    right_shoulder = landmarks[MP_LANDMARKS.RIGHT_SHOULDER]
    shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
    shoulder_midpoint = [shoulder_midpoint[0] * (x2 - x1), shoulder_midpoint[1] * (y2 - y1)]

    # Calculate the midpoint between the hips.
    left_hip = landmarks[MP_LANDMARKS.LEFT_HIP]
    right_hip = landmarks[MP_LANDMARKS.RIGHT_HIP]
    hip_midpoint = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
    hip_midpoint = [hip_midpoint[0] * (x2 - x1), hip_midpoint[1] * (y2 - y1)]

    toe_landmark = [landmarks[toe_idx].x * (x2 - x1), landmarks[toe_idx].y * (y2 - y1)]

    # Return the shoulder, toe, and hip landmarks along with all landmarks.
    return [shoulder_midpoint, toe_landmark, hip_midpoint], get_all_landmarks_coord(landmarks, get_boundaries(bounding_box))

def get_toe_tapping_signal(landmarks_list):
    """
    Generates a signal for the toe tapping task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - signal: The generated signal representing the toe movement.
    """
    signal = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, toe] = landmarks
        distance = math.dist(shoulder_midpoint, toe)  # Calculate the distance between the shoulder midpoint and toe.
        signal.append(distance)
    return signal

def get_toe_tapping_nf(landmarks_list):
    """
    Calculates the normalization factor for the toe tapping task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The average distance between the shoulder midpoint and hip midpoint, used as the normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, _, hip_midpoint] = landmarks
        distance = math.dist(shoulder_midpoint, hip_midpoint)  # Calculate the distance between shoulder and hip midpoints.
        values.append(distance)
    return np.mean(values)

def get_toe_tapping_display_landmarks(landmarks_list):
    """
    Selects landmarks for display during the toe tapping task.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The list of landmarks to be displayed.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, toe_landmark, _] = landmarks
        display_landmarks.append([shoulder_midpoint, toe_landmark])  # Add shoulder midpoint and toe landmark to display.
    return display_landmarks

def get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left):
    """
    Extracts the hand landmarks from the current frame using the bounding box and detector.

    Parameters:
    - bounding_box: The bounding box for the area of interest in the frame.
    - detector: The model used for detecting the landmarks.
    - current_frame_idx: The index of the current frame in the video.
    - current_frame: The current video frame.
    - is_left: Boolean indicating whether the task involves the left hand.

    Returns:
    - A list of detected hand landmarks.
    """
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    [x1, y1, x2, y2] = get_boundaries(bounding_box)

    image_data = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

    # Detect hand landmarks in the current frame.
    detection_result = detector.detect_for_video(image, current_frame_idx)
    current_frame_idx += 1

    hand_index = get_hand_index(detection_result, is_left)  # Determine which hand to track (left or right).

    if hand_index == -1:
        return []  # Return an empty list if the hand is not detected.

    return detection_result.hand_landmarks[hand_index]

def get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector):
    """
    Extracts essential landmarks for the hand movement task.

    Parameters:
    - current_frame: The current video frame.
    - current_frame_idx: The index of the current frame in the video.
    - bounding_box: The bounding box for the area of interest in the frame.
    - is_left: Boolean indicating whether the task involves the left hand.
    - detector: The model used for detecting the landmarks.

    Returns:
    - A tuple containing selected landmarks and all landmarks for the hand movement task.
    """
    hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)
    if not hand_landmarks:
        return [], []  # Return empty lists if no hand landmarks are detected.
    
    bounds = get_boundaries(bounding_box)
    index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
    middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
    ring_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP], bounds)
    wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

    if current_frame_idx == 6667:
        [x1, y1, x2, y2] = bounds
        landmarks = []
        landmarks.append(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP])
        landmarks.append(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP])
        landmarks.append(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP])
        # Save an image with hand landmarks for debugging or visualization.
        cv2.imwrite("outputs/" + str(current_frame_idx) + ".jpg", draw_hand(current_frame[y1:y2, x1:x2, :], landmarks))

    return [index_finger, middle_finger, ring_finger, wrist], get_all_landmarks_coord(hand_landmarks, bounds)

def get_hand_movement_signal(landmarks_list):
    """
    Generates a signal for the hand movement task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - signal: The generated signal representing the hand movement.
    """
    signal = []
    prevDist = 0
    for landmarks in landmarks_list:
        if len(landmarks) < 4:
            signal.append(prevDist)  # Use the previous distance if landmarks are insufficient.
            continue
        [index_finger, middle_finger, ring_finger, wrist] = landmarks
        # Calculate the average distance from the wrist to the fingers.
        distance = (math.dist(index_finger, wrist) + math.dist(middle_finger, wrist) + math.dist(ring_finger, wrist)) / 3
        prevDist = distance
        signal.append(distance)
    return signal

def get_hand_movement_nf(landmarks_list):
    """
    Calculates the normalization factor for the hand movement task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The maximum distance between the middle finger and wrist, used as the normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        if len(landmarks) < 4:
            continue  # Skip this iteration if landmarks length is less than 4.
        [_, middle_finger, _, wrist] = landmarks
        distance = math.dist(middle_finger, wrist)  # Calculate the distance between the middle finger and wrist.
        values.append(distance)
    return np.max(values)

def get_hand_movement_display_landmarks(landmarks_list):
    """
    Selects landmarks for display during the hand movement task.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The list of landmarks to be displayed.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        if len(landmarks) < 4:
            display_landmarks.append([])
            continue  # Skip this iteration if landmarks length is less than 4.
        [index_finger, middle_finger, ring_finger, wrist] = landmarks
        display_landmarks.append([index_finger, middle_finger, ring_finger, wrist])  # Add fingers and wrist to display.
    return display_landmarks

def get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector):
    """
    Extracts essential landmarks for the finger tap task.

    Parameters:
    - current_frame: The current video frame.
    - current_frame_idx: The index of the current frame in the video.
    - bounding_box: The bounding box for the area of interest in the frame.
    - is_left: Boolean indicating whether the task involves the left hand.
    - detector: The model used for detecting the landmarks.

    Returns:
    - A tuple containing selected landmarks and all landmarks for the finger tap task.
    """
    hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)
    if not hand_landmarks:
        return [], []  # Return empty lists if no hand landmarks are detected.

    bounds = get_boundaries(bounding_box)
    thumb_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP], bounds)
    index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
    middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
    wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

    if current_frame_idx == 1408:
        [x1, y1, x2, y2] = bounds
        landmarks = [hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP]]
        # Save an image with finger tap landmarks for debugging or visualization.
        cv2.imwrite("outputs/" + str(current_frame_idx) + ".jpg", draw_hand(current_frame[y1:y2, x1:x2, :], landmarks))

    return [thumb_finger, index_finger, middle_finger, wrist], get_all_landmarks_coord(hand_landmarks, bounds)

def get_finger_tap_signal(landmarks_list):
    """
    Generates a signal for the finger tap task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - signal: The generated signal representing the finger tap movement.
    """
    signal = []
    prev = 0
    for landmarks in landmarks_list:
        if not landmarks:
            signal.append(prev)  # Use the previous distance if landmarks are not detected.
            continue
        [thumb_finger, index_finger] = landmarks
        distance = math.dist(thumb_finger, index_finger)  # Calculate the distance between thumb and index finger.
        prev = distance
        signal.append(distance)
    return signal

def get_finger_tap_nf(landmarks_list):
    """
    Calculates the normalization factor for the finger tap task based on the landmarks.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The maximum distance between the middle finger and wrist, used as the normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        if not landmarks:
            continue
        [_, _, middle_finger, wrist] = landmarks
        distance = math.dist(middle_finger, wrist)  # Calculate the distance between the middle finger and wrist.
        values.append(distance)
    return np.max(values)

def get_finger_tap_display_landmarks(landmarks_list):
    """
    Selects landmarks for display during the finger tap task.

    Parameters:
    - landmarks_list: A list of landmarks for each frame.

    Returns:
    - The list of landmarks to be displayed.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        if not landmarks:
            display_landmarks.append([])
            continue
        [thumb_finger, index_finger, _, _] = landmarks
        display_landmarks.append([thumb_finger, index_finger])  # Add thumb and index finger to display.
    return display_landmarks

def get_hand_index(detection_result, is_left):
    """
    Determines the index of the hand (left or right) in the detection result.

    Parameters:
    - detection_result: The result from the hand detection model.
    - is_left: Boolean indicating whether to search for the left hand.

    Returns:
    - The index of the hand in the detection result, or -1 if not found.
    """
    direction = "Left" if is_left else "Right"

    handedness = detection_result.handedness

    for idx in range(0, len(handedness)):
        if handedness[idx][0].category_name == direction:
            return idx

    return -1  # Return -1 if the hand is not found.

def get_landmark_coords(landmark, bounds):
    """
    Converts normalized landmark coordinates to image coordinates.

    Parameters:
    - landmark: A normalized landmark from the detection result.
    - bounds: The bounding box for the area of interest.

    Returns:
    - A list containing the x and y coordinates of the landmark in image space.
    """
    [x1, y1, x2, y2] = bounds

    return [landmark.x * (x2 - x1), landmark.y * (y2 - y1)]

def get_all_landmarks_coord(landmark, bounds):
    """
    Converts all normalized landmarks to image coordinates.

    Parameters:
    - landmark: A list of normalized landmarks from the detection result.
    - bounds: The bounding box for the area of interest.

    Returns:
    - A list of lists, each containing the x, y, and z coordinates of a landmark in image space.
    """
    [x1, y1, x2, y2] = bounds
    return [[item.x * (x2-x1), item.y * (y2-y1), item.z] for item in landmark]

def get_all_landmarks_coord_YOLONAS(landmark):
    """
    Converts YOLO NAS landmarks to image coordinates.

    Parameters:
    - landmark: A list of landmarks from the YOLO NAS detection result.

    Returns:
    - A list of lists, each containing the x and y coordinates of a landmark.
    """
    return [[item[0], item[1]] for item in landmark]

def get_boundaries(bounding_box):
    """
    Calculates the boundaries of the bounding box in image space.

    Parameters:
    - bounding_box: The bounding box dictionary containing x, y, width, and height.

    Returns:
    - A list containing the x1, y1, x2, and y2 coordinates of the bounding box.
    """
    x1 = int(bounding_box['x'])
    y1 = int(bounding_box['y'])
    x2 = x1 + int(bounding_box['width'])
    y2 = y1 + int(bounding_box['height'])

    return [x1, y1, x2, y2]
