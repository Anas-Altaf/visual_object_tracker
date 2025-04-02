import os
import multiprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input paths: adjust these paths as needed
VIDEO_PATH = os.path.join(BASE_DIR, "input_video.mp4")
REFERENCE_IMAGE_PATH = os.path.join(BASE_DIR, "reference_image.jpg")

# Output path for processed video
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output_video.mp4")

# Processing parameters
SIMILARITY_THRESHOLD = 0.4  # Minimum similarity score to consider a match (0-1)
FRAME_SKIP = 1  # Process every Nth frame (1 = process all frames)
DISPLAY_PROGRESS = True  # Show progress bar
RESIZE_FACTOR = 0.5  # Resize factor for processing (smaller = faster)
NUM_SEGMENTS = max(1, min(multiprocessing.cpu_count(), 8))  # Use available CPU cores up to 8
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # Face detection confidence threshold for video frames
REFERENCE_CONFIDENCE_THRESHOLD = 0.3  # Face detection confidence threshold for reference image
