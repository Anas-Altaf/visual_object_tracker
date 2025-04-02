import os
from modules import optimized_process_video2
import config

def main():
    result = optimized_process_video2(
        video_path=config.VIDEO_PATH,
        reference_path=config.REFERENCE_IMAGE_PATH,
        output_path=config.OUTPUT_VIDEO_PATH,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        frame_skip=config.FRAME_SKIP,
        display_progress=config.DISPLAY_PROGRESS,
        resize_factor=config.RESIZE_FACTOR,
        num_segments=config.NUM_SEGMENTS,
        detection_conf_threshold=config.DETECTION_CONFIDENCE_THRESHOLD,
        reference_conf_threshold=config.REFERENCE_CONFIDENCE_THRESHOLD
    )
    print(result)

if __name__ == "__main__":
    main()
