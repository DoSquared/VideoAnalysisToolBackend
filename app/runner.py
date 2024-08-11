from app.analysis.YOLOTracker import YOLOTracker  # Import the YOLOTracker class from the app's analysis module.
import json  # Import the JSON module to handle JSON data.
import time  # Import the time module to measure execution time.

# Function to write the output dictionary to a file in JSON format.
def write_output_to_file(output, file_path):
    # Open the specified file path in write mode.
    with open(file_path, 'w') as outfile:
        # Serialize the output dictionary to JSON and write it to the file.
        json.dump(output, outfile)

# Record the start time to measure how long the YOLOTracker process takes.
start_time = time.time()

# Run the YOLOTracker on the specified video file and model, storing the results in 'ouputDict'.
ouputDict = YOLOTracker("rigidity_gaby.mp4", 'yolov8n.pt', '')

# Print the time taken to complete the YOLOTracker process.
print("--- %s seconds ---" % (time.time() - start_time))

# Write the YOLOTracker output dictionary to a JSON file named 'output.json'.
write_output_to_file(ouputDict, 'output.json')
