import numpy as np
import scipy.signal as signal

def compareNeighboursNegative(item1, item2, distance, minDistance=5):
    """
    Compares two consecutive elements (item1 and item2) from a list where velocities are negative,
    and determines if one should be kept or modified based on their proximity and speed.

    Parameters:
    - item1, item2: Dictionary objects containing indices and speeds of peaks and valleys.
    - distance: An array representing the distance signal.
    - minDistance: Minimum distance between peaks and valleys for them to be considered distinct.

    Returns:
    - newItem: A modified or selected dictionary item based on the comparison, or None if no modification is needed.
    """
    # Case 1: The peak of item1 and the valley of item2 are too close.
    if abs(item1['valleyIndex'] - item2['peakIndex']) < minDistance:
        # Keep the item with the highest speed and combine peak and valley indices.
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {
                'maxSpeedIndex': item1['maxSpeedIndex'],
                'maxSpeed': item1['maxSpeed'],
                'peakIndex': item1['peakIndex'],
                'valleyIndex': item2['valleyIndex']
            }
        else:
            newItem = {
                'maxSpeedIndex': item2['maxSpeedIndex'],
                'maxSpeed': item2['maxSpeed'],
                'peakIndex': item1['peakIndex'],
                'valleyIndex': item2['valleyIndex']
            }
        return newItem

    # Case 2: The peaks of both items are too close.
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # Keep the item with the highest speed.
        newItem = item1 if item1['maxSpeed'] > item2['maxSpeed'] else item2
        return newItem

    # Case 3: The valleys of both items are too close.
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # Keep the item with the highest speed.
        newItem = item1 if item1['maxSpeed'] > item2['maxSpeed'] else item2
        return newItem

    # Case 4: The valley of item1 is of similar height to the peak of item2.
    if abs(distance[item1['valleyIndex']] - distance[item2['peakIndex']]) < abs(
            distance[item1['valleyIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # Keep the item with the highest speed and combine peak and valley indices.
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {
                'maxSpeedIndex': item1['maxSpeedIndex'],
                'maxSpeed': item1['maxSpeed'],
                'peakIndex': item1['peakIndex'],
                'valleyIndex': item2['valleyIndex']
            }
        else:
            newItem = {
                'maxSpeedIndex': item2['maxSpeedIndex'],
                'maxSpeed': item2['maxSpeed'],
                'peakIndex': item1['peakIndex'],
                'valleyIndex': item2['valleyIndex']
            }
        return newItem

    return None

def compareNeighboursPositive(item1, item2, distance, minDistance=5):
    """
    Similar to compareNeighboursNegative, but for positive velocities. Compares two consecutive elements 
    and decides whether to keep, modify, or combine them based on their proximity and speed.

    Parameters:
    - item1, item2: Dictionary objects containing indices and speeds of peaks and valleys.
    - distance: An array representing the distance signal.
    - minDistance: Minimum distance between peaks and valleys for them to be considered distinct.

    Returns:
    - newItem: A modified or selected dictionary item based on the comparison, or None if no modification is needed.
    """
    # Case 1: The peak of item1 and the valley of item2 are too close.
    if abs(item1['peakIndex'] - item2['valleyIndex']) < minDistance:
        # Keep the item with the highest speed and combine peak and valley indices.
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {
                'maxSpeedIndex': item1['maxSpeedIndex'],
                'maxSpeed': item1['maxSpeed'],
                'peakIndex': item2['peakIndex'],
                'valleyIndex': item1['valleyIndex']
            }
        else:
            newItem = {
                'maxSpeedIndex': item2['maxSpeedIndex'],
                'maxSpeed': item2['maxSpeed'],
                'peakIndex': item2['peakIndex'],
                'valleyIndex': item1['valleyIndex']
            }
        return newItem

    # Case 2: The peaks of both items are too close.
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # Keep the item with the highest speed.
        newItem = item1 if item1['maxSpeed'] > item2['maxSpeed'] else item2
        return newItem

    # Case 3: The valleys of both items are too close.
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # Keep the item with the highest speed.
        newItem = item1 if item1['maxSpeed'] > item2['maxSpeed'] else item2
        return newItem

    # Case 4: The peak of item1 is of similar height to the valley of item2.
    if abs(distance[item1['peakIndex']] - distance[item2['valleyIndex']]) < abs(
            distance[item1['peakIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # Keep the item with the highest speed and combine peak and valley indices.
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {
                'maxSpeedIndex': item1['maxSpeedIndex'],
                'maxSpeed': item1['maxSpeed'],
                'peakIndex': item2['peakIndex'],
                'valleyIndex': item1['valleyIndex']
            }
        else:
            newItem = {
                'maxSpeedIndex': item2['maxSpeedIndex'],
                'maxSpeed': item2['maxSpeed'],
                'peakIndex': item2['peakIndex'],
                'valleyIndex': item1['valleyIndex']
            }
        return newItem

    return None

def eliminateBadNeighboursNegative(indexVelocity, distance, minDistance=5):
    """
    Eliminates closely spaced negative velocity peaks and valleys that may indicate noise
    or incorrect detection. This is done by comparing consecutive items and keeping only the most significant one.

    Parameters:
    - indexVelocity: List of dictionary items representing the peaks and valleys of negative velocities.
    - distance: An array representing the distance signal.
    - minDistance: Minimum distance between peaks and valleys for them to be considered distinct.

    Returns:
    - indexVelocityCorrected: A list of corrected dictionary items after eliminating bad neighbours.
    """
    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)  # Initialize a list to track items to be skipped.

    for idx in range(len(indexVelocity)):
        if isSkip[idx] == False:  # Process only if the item is not marked for skipping.
            if idx < len(indexVelocity) - 1:
                # Compare current item with the next one.
                newItem = compareNeighboursNegative(indexVelocity[idx], indexVelocity[idx + 1], distance, minDistance)
                if newItem is not None:
                    # If a new item is created by comparison, add it to the corrected list and skip the next item.
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # If no modification is needed, keep the current item.
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                # For the last item in the list, just add it to the corrected list.
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected

def eliminateBadNeighboursPositive(indexVelocity, distance, minDistance=5):
    """
    Similar to eliminateBadNeighboursNegative, but for positive velocities. 
    Eliminates closely spaced positive velocity peaks and valleys.

    Parameters:
    - indexVelocity: List of dictionary items representing the peaks and valleys of positive velocities.
    - distance: An array representing the distance signal.
    - minDistance: Minimum distance between peaks and valleys for them to be considered distinct.

    Returns:
    - indexVelocityCorrected: A list of corrected dictionary items after eliminating bad neighbours.
    """
    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)  # Initialize a list to track items to be skipped.

    for idx in range(len(indexVelocity)):
        if isSkip[idx] == False:  # Process only if the item is not marked for skipping.
            if idx < len(indexVelocity) - 1:
                # Compare current item with the next one.
                newItem = compareNeighboursPositive(indexVelocity[idx], indexVelocity[idx + 1], distance, minDistance=minDistance)
                if newItem is not None:
                    # If a new item is created by comparison, add it to the corrected list and skip the next item.
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # If no modification is needed, keep the current item.
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                # For the last item in the list, just add it to the corrected list.
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected

def correctBasedonHeight(pos, distance, prct=0.125, minDistance=5):
    """
    Filters out peaks that are smaller than a certain percentage of the average peak height, 
    as they may indicate noise or insignificant features.

    Parameters:
    - pos: List of dictionary items representing peaks and valleys.
    - distance: An array representing the distance signal.
    - prct: The percentage of the average height below which peaks will be filtered out.
    - minDistance: Minimum distance between peaks and valleys for them to be considered distinct.

    Returns:
    - corrected: A list of corrected dictionary items after filtering based on height.
    """
    heightPeaks = []
    for item in pos:
        try:
            heightPeaks.append(abs(distance[item['peakIndex']] - distance[item['valleyIndex']]))
        except:
            pass

    meanHeightPeak = np.mean(heightPeaks)  # Calculate the mean height of the peaks.
    corrected = []
    for item in pos:
        try:
            # Keep only peaks that are above the specified percentage of the mean height and have valid distances.
            if (abs(distance[item['peakIndex']] - distance[item['valleyIndex']])) > prct * meanHeightPeak:
                if abs(item['peakIndex'] - item['valleyIndex']) >= minDistance:
                    if (distance[item['peakIndex']] > distance[item['maxSpeedIndex']]) and (
                            distance[item['valleyIndex']] < distance[item['maxSpeedIndex']]):
                        corrected.append(item)
        except:
            pass

    return corrected

def correctBasedonVelocityNegative(pos, velocity, prct=0.125):
    """
    Filters out negative velocity peaks that are below a certain percentage of the average peak velocity.

    Parameters:
    - pos: List of dictionary items representing peaks and valleys.
    - velocity: An array representing the velocity signal.
    - prct: The percentage of the average velocity below which peaks will be filtered out.

    Returns:
    - corrected: A list of corrected dictionary items after filtering based on velocity.
    """
    velocity = velocity ** 2  # Square the velocity to emphasize larger peaks.

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)  # Calculate the mean velocity of the peaks.
    corrected = []
    for item in pos:
        try:
            # Keep only peaks that are above the specified percentage of the mean velocity.
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
        except:
            pass

    return corrected

def correctBasedonVelocityPositive(pos, velocity, prct=0.125):
    """
    Similar to correctBasedonVelocityNegative, but for positive velocities. 
    Filters out positive velocity peaks that are below a certain percentage of the average peak velocity.

    Parameters:
    - pos: List of dictionary items representing peaks and valleys.
    - velocity: An array representing the velocity signal.
    - prct: The percentage of the average velocity below which peaks will be filtered out.

    Returns:
    - corrected: A list of corrected dictionary items after filtering based on velocity.
    """
    velocity[velocity < 0] = 0  # Remove negative velocities.
    velocity = velocity ** 2  # Square the velocity to emphasize larger peaks.

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)  # Calculate the mean velocity of the peaks.
    corrected = []
    for item in pos:
        try:
            # Keep only peaks that are above the specified percentage of the mean velocity.
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
        except:
            pass

    return corrected

def correctFullPeaks(distance, pos, neg):
    """
    Matches positive and negative velocity peaks to identify full peaks, 
    which represent significant events in the signal.

    Parameters:
    - distance: An array representing the distance signal.
    - pos: List of dictionary items representing positive peaks and valleys.
    - neg: List of dictionary items representing negative peaks and valleys.

    Returns:
    - peakCandidatesCorrected: A list of corrected dictionary items representing full peaks.
    """
    closingVelocities = [item['maxSpeedIndex'] for item in neg]  # Indices of negative velocity peaks.
    openingVelocities = [item['maxSpeedIndex'] for item in pos]  # Indices of positive velocity peaks.

    peakCandidates = []
    for idx, closingVelocity in enumerate(closingVelocities):
        try:
            difference = np.array(openingVelocities) - closingVelocity  # Calculate differences between closing and opening velocities.
            difference[difference > 0] = 0

            posmin = np.argmax(difference[np.nonzero(difference)])  # Find the most significant match.

            # Identify the absolute peak within the range of interest.
            absolutePeak = np.max(distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])
            absolutePeakIndex = pos[posmin]['maxSpeedIndex'] + np.argmax(
                distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])

            # Create a new peak candidate.
            peakCandidate = {
                'openingValleyIndex': pos[posmin]['valleyIndex'],
                'openingPeakIndex': pos[posmin]['peakIndex'],
                'openingMaxSpeedIndex': pos[posmin]['maxSpeedIndex'],
                'closingValleyIndex': neg[idx]['valleyIndex'],
                'closingPeakIndex': neg[idx]['peakIndex'],
                'closingMaxSpeedIndex': neg[idx]['maxSpeedIndex'],
                'peakIndex': absolutePeakIndex
            }

            peakCandidates.append(peakCandidate)
        except:
            pass

    # Correct for overlapping peaks and valleys.
    peakCandidatesCorrected = []
    idx = 0
    while idx < len(peakCandidates):
        peakCandidate = peakCandidates[idx]
        peak = peakCandidate['peakIndex']
        difference = [(peak - item['peakIndex']) for item in peakCandidates]
        if len(np.where(np.array(difference) == 0)[0]) == 1:
            peakCandidatesCorrected.append(peakCandidate)
            idx += 1
        else:
            # Merge overlapping candidates.
            item1 = peakCandidates[np.where(np.array(difference) == 0)[0][0]]
            item2 = peakCandidates[np.where(np.array(difference) == 0)[0][1]]
            peakCandidate = {
                'openingValleyIndex': item1['openingValleyIndex'],
                'openingPeakIndex': item1['openingPeakIndex'],
                'openingMaxSpeedIndex': item1['openingMaxSpeedIndex'],
                'closingValleyIndex': item2['closingValleyIndex'],
                'closingPeakIndex': item2['closingPeakIndex'],
                'closingMaxSpeedIndex': item2['closingMaxSpeedIndex'],
                'peakIndex': item2['peakIndex']
            }
            peakCandidatesCorrected.append(peakCandidate)
            idx += 2

    return peakCandidatesCorrected

def correctBasedonPeakSymmetry(peaks):
    """
    Filters out peaks that do not exhibit symmetry between their opening and closing valleys, 
    as asymmetric peaks may indicate noise or insignificant features.

    Parameters:
    - peaks: List of dictionary items representing peaks.

    Returns:
    - peaksCorrected: A list of corrected dictionary items after filtering based on peak symmetry.
    """
    peaksCorrected = []
    for peak in peaks:
        leftValley = peak['openingValleyIndex']
        centerPeak = peak['peakIndex']
        rightValley = peak['closingValleyIndex']

        # Calculate the ratio of the distances from the peak to the valleys.
        ratio = (centerPeak - leftValley) / (rightValley - centerPeak)
        if 0.25 <= ratio <= 4:  # Keep only peaks with a reasonable symmetry ratio.
            peaksCorrected.append(peak)

    return peaksCorrected

def peakFinder(rawSignal, fs=30, minDistance=5, cutOffFrequency=5, prct=0.125):
    """
    Detects significant peaks in a raw signal based on velocity, height, and symmetry criteria. 
    Applies filtering and correction steps to remove noise and incorrect detections.

    Parameters:
    - rawSignal: The input signal from which to detect peaks.
    - fs: Sampling frequency of the input signal.
    - minDistance: Minimum distance between peaks and valleys for them to be considered distinct.
    - cutOffFrequency: Cut-off frequency for the low-pass filter applied to the signal.
    - prct: The percentage of the average peak height/velocity below which peaks will be filtered out.

    Returns:
    - distance: The filtered signal representing distance.
    - velocity: The derived velocity from the filtered signal.
    - peaks: The final list of detected peaks after all correction steps.
    - indexPositiveVelocity: The list of detected positive velocity peaks before correction.
    - indexNegativeVelocity: The list of detected negative velocity peaks before correction.
    """
    indexPositiveVelocity = []
    indexNegativeVelocity = []

    # Apply a low-pass filter to the signal.
    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='low', analog=False)
    distance = signal.filtfilt(b, a, rawSignal)  # Filter the signal to get the distance.
    velocity = signal.savgol_filter(distance, 5, 3, deriv=1) / (1 / fs)  # Compute the velocity from the distance.

    # Approximate mean frequency using autocorrelation.
    acorr = np.convolve(rawSignal, rawSignal)
    t0 = ((1 / fs) * np.argmax(acorr))
    sep = 0.5 * (t0) if (0.5 * t0 > 1) else 1  # Define minimum separation between peaks.

    deriv = velocity.copy()
    deriv[deriv < 0] = 0  # Consider only positive velocities.
    deriv = deriv ** 2  # Square the derivative to emphasize larger peaks.

    peaks, props = signal.find_peaks(deriv, distance=sep)  # Find positive velocity peaks.

    heightPeaksPositive = deriv[peaks]  # Heights of the detected positive peaks.
    selectedPeaksPositive = peaks[heightPeaksPositive > prct * np.mean(heightPeaksPositive)]  # Filter out small peaks.

    # Identify peaks and valleys for each detected positive peak.
    for idx, peak in enumerate(selectedPeaksPositive):
        idxValley = peak - 1
        if idxValley >= 0:
            while deriv[idxValley] != 0:
                if idxValley <= 0:
                    idxValley = np.nan
                    break
                idxValley -= 1

        idxPeak = peak + 1
        if idxPeak < len(deriv):
            while deriv[idxPeak] != 0:
                if idxPeak >= len(deriv) - 1:
                    idxPeak = np.nan
                    break
                idxPeak += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            positiveVelocity = {
                'maxSpeedIndex': peak,
                'maxSpeed': np.sqrt(deriv[peak]),
                'peakIndex': idxPeak,
                'valleyIndex': idxValley
            }
            indexPositiveVelocity.append(positiveVelocity)

    deriv = velocity.copy()
    deriv[deriv > 0] = 0  # Consider only negative velocities.
    deriv = deriv ** 2  # Square the derivative to emphasize larger peaks.
    peaks, props = signal.find_peaks(deriv, distance=sep)  # Find negative velocity peaks.

    heightPeaksNegative = deriv[peaks]  # Heights of the detected negative peaks.
    selectedPeaksNegative = peaks[heightPeaksNegative > prct * np.mean(heightPeaksNegative)]  # Filter out small peaks.

    # Identify peaks and valleys for each detected negative peak.
    for idx, peak in enumerate(selectedPeaksNegative):
        idxPeak = peak - 1
        if idxPeak >= 0:
            while deriv[idxPeak] != 0:
                if idxPeak <= 0:
                    idxPeak = np.nan
                    break
                idxPeak -= 1

        idxValley = peak + 1
        if idxValley < len(deriv):
            while deriv[idxValley] != 0:
                if idxValley >= len(deriv) - 1:
                    idxValley = np.nan
                    break
                idxValley += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            negativeVelocity = {
                'maxSpeedIndex': peak,
                'maxSpeed': np.sqrt(deriv[peak]),
                'peakIndex': idxPeak,
                'valleyIndex': idxValley
            }
            indexNegativeVelocity.append(negativeVelocity)

    # Eliminate bad peaks through a series of corrections.
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    indexPositiveVelocity = correctBasedonHeight(indexPositiveVelocity, distance)
    indexPositiveVelocity = correctBasedonVelocityPositive(indexPositiveVelocity, velocity.copy())

    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    indexNegativeVelocity = correctBasedonHeight(indexNegativeVelocity, distance)
    indexNegativeVelocity = correctBasedonVelocityNegative(indexNegativeVelocity, velocity.copy())

    peaks = correctFullPeaks(distance, indexPositiveVelocity, indexNegativeVelocity)
    peaks = correctBasedonPeakSymmetry(peaks)

    return distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity
