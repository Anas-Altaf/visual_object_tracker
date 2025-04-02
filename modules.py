import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
import threading
import time
import random
from tqdm.notebook import tqdm
from scipy.spatial.distance import cosine

# --------------------------
# Face Detection Setup - OpenCV DNN
# --------------------------
def load_opencv_face_detector():
    """
    Load OpenCV's DNN-based face detector (ResNet10)
    """
    try:
        model_file = os.path.join(os.getcwd(), "opencv_face_detector_uint8.pb")
        config_file = os.path.join(os.getcwd(), "opencv_face_detector.pbtxt")
        if not os.path.isfile(model_file) or not os.path.isfile(config_file):
            print(f"Downloading face detector model files...")
            import urllib.request
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb",
                model_file
            )
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/8b4b382fa42016dc04a4a9247425bdc8525cbb94/samples/dnn/face_detector/opencv_face_detector.pbtxt",
                config_file
            )
            print(f"Model files downloaded to {os.getcwd()}")
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        print("OpenCV DNN face detector loaded successfully.")
        return net
    except Exception as e:
        print(f"Error loading OpenCV DNN face detector: {str(e)}")
        print("Falling back to Haar Cascade face detector...")
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        print(f"Loading Haar cascade from: {haar_file}")
        cascade = cv2.CascadeClassifier(haar_file)
        if cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        return cascade

face_detector = load_opencv_face_detector()

def detect_faces(image, confidence_threshold=0.5):
    """
    Detect faces using OpenCV's DNN face detector
    
    Args:
        image: RGB numpy array image
        confidence_threshold: Minimum detection confidence (0-1)
    """
    if image is None or image.size == 0:
        print("Warning: Empty image passed to detect_faces")
        return []
    
    h, w = image.shape[:2]
    
    if isinstance(face_detector, cv2.dnn.Net):
        try:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
            face_detector.setInput(blob)
            detections = face_detector.forward()
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    width, height = x2 - x1, y2 - y1
                    if width <= 0 or height <= 0:
                        continue
                    faces.append({
                        "box": [x1, y1, width, height],
                        "keypoints": {},
                        "confidence": float(confidence)
                    })
            if not faces and confidence_threshold > 0.2:
                return detect_faces(image, confidence_threshold=0.2)
            return faces
        except Exception as e:
            print(f"Error in DNN face detection: {str(e)}")
            print("Falling back to Haar Cascade due to DNN error")
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                detections = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                faces = []
                for (x, y, w_val, h_val) in detections:
                    faces.append({
                        "box": [int(x), int(y), int(w_val), int(h_val)],
                        "keypoints": {},
                        "confidence": 1.0
                    })
                return faces
            except Exception as e:
                print(f"Error in Haar Cascade fallback: {str(e)}")
                return []
    else:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if gray.size == 0 or gray is None:
                print("Invalid grayscale image for Haar detection")
                return []
            detections = face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = []
            if len(detections) == 0:
                return []
            for detection in detections:
                if len(detection) == 4:
                    x, y, w_val, h_val = detection
                    faces.append({
                        "box": [int(x), int(y), int(w_val), int(h_val)],
                        "keypoints": {},
                        "confidence": 1.0
                    })
            return faces
        except Exception as e:
            print(f"Error in Haar Cascade face detection: {str(e)}")
            return []

# --------------------------
# Visualization utilities
# --------------------------
MARGIN = 10  
ROW_SIZE = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  

def visualize(image, detections):
    """
    Visualize face detections on an image
    """
    annotated_image = image.copy()
    if not detections:
        return annotated_image
    for face in detections:
        x, y, w_val, h_val = face["box"]
        cv2.rectangle(annotated_image, (int(x), int(y)), (int(x+w_val), int(y+h_val)), TEXT_COLOR, 3)
        if "keypoints" in face and face["keypoints"]:
            landmarks = face["keypoints"]
            for point in landmarks.values():
                cv2.circle(annotated_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), 2)
        if "confidence" in face:
            probability = round(face["confidence"], 2)
            result_text = f'Face ({probability})'
            text_location = (MARGIN + int(x), MARGIN + ROW_SIZE + int(y))
            cv2.putText(annotated_image, result_text, text_location,
                        cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return annotated_image

# --------------------------
# Module 1: Image Loading
# --------------------------
def load_image(image_path):
    """
    Load and convert an image from BGR to RGB
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

# --------------------------
# Module 3: Face Extraction and Alignment
# --------------------------
def extract_face(image, face_info, required_size=(160, 160)):
    """
    Extract and resize a face from an image
    """
    try:
        x, y, width, height = face_info['box']
        x, y = max(0, x), max(0, y)
        x_end = min(x + width, image.shape[1])
        y_end = min(y + height, image.shape[0])
        if x_end <= x or y_end <= y:
            print("Warning: Invalid face box dimensions")
            return None, None
        face = image[y:y_end, x:x_end]
        if face.size == 0:
            print("Warning: Extracted face has zero size")
            return None, None
        face_image = cv2.resize(face, required_size)
        face_image = face_image.astype('float32') / 255.0
        return face_image, (x, y, width, height)
    except Exception as e:
        print(f"Error extracting face: {e}")
        return None, None

def align_face(img, landmarks):
    """
    Align face based on eye landmarks if available
    """
    try:
        if not landmarks or 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return img
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        return aligned
    except Exception as e:
        print(f"Warning: Face alignment failed: {e}")
        return img

# --------------------------
# Module 4: Face Embedding and Matching (Using FaceNet via facenet-pytorch)
# --------------------------
def get_embedding(face_image, model):
    """
    Generate embedding for a face image using a pre-trained model
    """
    try:
        if face_image is None or face_image.size == 0:
            print("Warning: Invalid face image passed to get_embedding")
            return None
        if face_image.shape[2] == 4:
            face_image = face_image[:, :, :3]
        face_image = (face_image * 2) - 1
        embedding = model.predict(face_image)[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    """
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0

def match_embeddings(ref_embedding, detection_embedding, threshold=0.8):
    """
    Match two face embeddings based on cosine similarity
    """
    try:
        if ref_embedding is None or detection_embedding is None:
            return False, 0
        sim = cosine_similarity(ref_embedding, detection_embedding)
        return sim >= threshold, sim
    except Exception as e:
        print(f"Error matching embeddings: {e}")
        return False, 0

def load_facenet_model():
    """
    Load pre-trained FaceNet model
    """
    print("Loading VGGFace2 model from facenet-pytorch...")
    try:
        try:
            import facenet_pytorch
        except ImportError:
            print("Installing facenet-pytorch...")
            import subprocess
            subprocess.check_call(["pip", "install", "facenet-pytorch"])
            import facenet_pytorch
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='vggface2').eval()
        import torch
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Model moved to GPU.")
        def predict_wrapper(img_array):
            import torch
            try:
                if isinstance(img_array, np.ndarray):
                    if len(img_array.shape) == 3:
                        img_array = np.expand_dims(img_array, axis=0)
                    tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
                    if torch.cuda.is_available():
                        tensor = tensor.to('cuda')
                    with torch.no_grad():
                        embedding = model(tensor).cpu().numpy()
                    return embedding
                else:
                    raise ValueError("Input must be a numpy array")
            except Exception as e:
                print(f"Error in prediction: {e}")
                return None
        model.predict = predict_wrapper
        print("VGGFace2 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

# --------------------------
# Progress Counter for Video Processing
# --------------------------
class ProgressCounter:
    def __init__(self, total, pbar=None):
        self.counter = 0
        self.total = total
        self.pbar = pbar
        self.lock = threading.Lock()
        
    def update(self, amount=1, desc=None):
        with self.lock:
            self.counter += amount
            if self.pbar:
                self.pbar.update(amount)
                if desc:
                    self.pbar.set_description(desc)

# --------------------------
# Video Processing: Optimized Function
# --------------------------
def optimized_process_video2(video_path, reference_path, output_path, 
                             similarity_threshold=0.4, frame_skip=1, 
                             display_progress=True, resize_factor=0.5,
                             num_segments=4, detection_conf_threshold=0.5,
                             reference_conf_threshold=0.3):
    """
    Process video using face detection and optimized processing.
    
    Args:
        video_path: Path to input video
        reference_path: Path to reference image
        output_path: Path for output video
        similarity_threshold: Minimum similarity score to consider a match (0-1)
        frame_skip: Process every Nth frame
        display_progress: Show progress bar
        resize_factor: Scale factor for processing (smaller = faster)
        num_segments: Number of parallel processing segments
        detection_conf_threshold: Face detection confidence threshold for video frames
        reference_conf_threshold: Face detection confidence threshold for reference image
    """
    try:
        # Initial setup and validation
        if not os.path.exists(video_path):
            return {"success": False, "message": f"Video file not found: {video_path}"}
        if not os.path.exists(reference_path):
            return {"success": False, "message": f"Reference image not found: {reference_path}"}
        
        # Process reference image
        print(f"Loading reference image from: {reference_path}")
        reference_image = load_image(reference_path)
        print(f"Reference image loaded successfully. Shape: {reference_image.shape}")
        
        debug_ref_path = "debug_reference.jpg"
        cv2.imwrite(debug_ref_path, cv2.cvtColor(reference_image, cv2.COLOR_RGB2BGR))
        print(f"Saved debug reference image to: {debug_ref_path}")
        
        print("Detecting faces in reference image...")
        reference_faces = detect_faces(reference_image, confidence_threshold=reference_conf_threshold)
        print(f"Found {len(reference_faces)} faces in reference image.")
        
        if len(reference_faces) > 0:
            debug_vis_path = "debug_reference_with_faces.jpg"
            vis_img = visualize(reference_image, reference_faces)
            cv2.imwrite(debug_vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization with detections to: {debug_vis_path}")
            
        if not reference_faces:
            print("WARNING: No faces detected in reference image. This will cause the process to fail.")
            return {"success": False, "message": "No face detected in reference image"}
        
        # Load face embedding model - this will be shared across threads
        model = load_facenet_model()
        
        # Extract reference face embedding
        ref_face_info = reference_faces[0]
        ref_face, ref_box = extract_face(reference_image, ref_face_info)
        
        if ref_face is None:
            return {"success": False, "message": "Failed to extract face from reference image"}
            
        debug_face_path = "debug_reference_face.jpg"
        cv2.imwrite(debug_face_path, cv2.cvtColor((ref_face * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"Saved extracted reference face to: {debug_face_path}")
        
        if ref_face_info['keypoints']:
            ref_face = align_face(ref_face, ref_face_info['keypoints'])
            
        ref_embedding = get_embedding(ref_face, model)
        
        if ref_embedding is None:
            return {"success": False, "message": "Failed to generate embedding for reference face"}
            
        print(f"Reference embedding computed successfully. Shape: {ref_embedding.shape}")
        
        # Get video parameters
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "message": f"Could not open video file: {video_path}"}
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        process_width = int(width * resize_factor)
        process_height = int(height * resize_factor)
        
        # Create segments for parallel processing
        segments = []
        frames_per_segment = max(1, total_frames // num_segments)
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = (i + 1) * frames_per_segment if i < num_segments - 1 else total_frames
            segments.append((i, start_frame, end_frame))
        
        print(f"Dividing video into {len(segments)} segments for parallel processing")
        for idx, (segment_id, start, end) in enumerate(segments):
            print(f"  Segment {segment_id}: frames {start} to {end} ({end-start} frames)")
            
        total_ops = total_frames * 2  # Each frame is processed twice (detection + embedding)
        pbar = tqdm(total=total_ops, desc="Initializing...") if display_progress else None
        progress = ProgressCounter(total_ops, pbar)
        
        # Thread-local storage to prevent race conditions with OpenCV
        thread_local = threading.local()
        
        def get_video_capture():
            """Get thread-local video capture object"""
            if not hasattr(thread_local, "video_capture"):
                thread_local.video_capture = cv2.VideoCapture(video_path)
                if not thread_local.video_capture.isOpened():
                    raise IOError(f"Thread failed to open video: {video_path}")
            return thread_local.video_capture
            
        def process_segment(segment_info):
            segment_id, start_frame, end_frame = segment_info
            try:
                # Get a thread-local video capture instance
                cap = get_video_capture()
                
                # Set video position to segment start
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Initialize tracking variables
                frame_idx = start_frame
                segment_frames = {}
                segment_matches = 0
                frames_processed = 0
                
                # Process all frames in this segment
                while frame_idx < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Segment {segment_id}: Failed to read frame {frame_idx}, stopping")
                        break
                        
                    progress.update(1, f"Segment {segment_id}: {frames_processed}/{end_frame-start_frame}")
                    
                    # Only process every Nth frame based on frame_skip
                    if frame_idx % frame_skip == 0:
                        try:
                            # Resize for faster processing
                            resized = cv2.resize(frame, (process_width, process_height))
                            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            
                            # Detect faces with the configured threshold
                            faces = detect_faces(rgb, confidence_threshold=detection_conf_threshold)
                            
                            # Process each detected face
                            for face in faces:
                                # Scale coordinates back to original size
                                x, y, w_val, h_val = face['box']
                                x_orig = int(x / resize_factor)
                                y_orig = int(y / resize_factor)
                                w_orig = int(w_val / resize_factor)
                                h_orig = int(h_val / resize_factor)
                                
                                # Ensure coordinates are within bounds
                                x_orig = max(0, x_orig)
                                y_orig = max(0, y_orig)
                                w_orig = min(w_orig, frame.shape[1] - x_orig)
                                h_orig = min(h_orig, frame.shape[0] - y_orig)
                                
                                if w_orig <= 0 or h_orig <= 0:
                                    continue
                                    
                                # Draw initial rectangle for all detected faces (blue)
                                cv2.rectangle(frame, (x_orig, y_orig), 
                                              (x_orig + w_orig, y_orig + h_orig), 
                                              (255, 0, 0), 2)
                                              
                                # Extract face for embedding comparison
                                try:
                                    face_img = frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
                                    if face_img.size == 0:
                                        continue
                                        
                                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                    face_resized = cv2.resize(face_rgb, (160, 160))
                                    face_normalized = face_resized.astype('float32') / 255.0
                                    
                                    # Align if landmarks available
                                    if face['keypoints']:
                                        face_normalized = align_face(face_normalized, face['keypoints'])
                                    
                                    # Get face embedding
                                    embedding = get_embedding(face_normalized, model)
                                    
                                    # Compare with reference face
                                    if embedding is not None:
                                        similarity = 1 - cosine(ref_embedding, embedding)
                                        
                                        # Mark matches with green rectangle
                                        if similarity > similarity_threshold:
                                            cv2.rectangle(frame, (x_orig, y_orig), 
                                                          (x_orig + w_orig, y_orig + h_orig), 
                                                          (0, 255, 0), 3)
                                            cv2.putText(frame, f"{similarity:.2f}", 
                                                        (x_orig, y_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                        0.9, (0, 255, 0), 2)
                                            segment_matches += 1
                                except Exception as e:
                                    print(f"Error processing face in frame {frame_idx}: {e}")
                                    continue
                            
                            # Update progress a second time for embedding generation
                            progress.update(1, f"Segment {segment_id}: {frames_processed}/{end_frame-start_frame}")
                            
                        except Exception as e:
                            print(f"Error processing frame {frame_idx} in segment {segment_id}: {e}")
                    
                    # Store processed frame
                    segment_frames[frame_idx] = frame
                    frame_idx += 1
                    frames_processed += 1
                
                # Thread is done with its video capture
                # Don't close it as that can cause issues with thread-local storage
                
                print(f"Segment {segment_id} complete: processed {frames_processed} frames, found {segment_matches} matches")
                
                return {
                    "segment_id": segment_id, 
                    "frames": segment_frames, 
                    "matches": segment_matches,
                    "start_frame": start_frame, 
                    "end_frame": end_frame,
                    "frames_processed": frames_processed
                }
                
            except Exception as e:
                import traceback
                print(f"Error in segment {segment_id}: {e}")
                print(traceback.format_exc())
                return {
                    "segment_id": segment_id, 
                    "frames": {}, 
                    "matches": 0,
                    "start_frame": start_frame, 
                    "end_frame": end_frame,
                    "error": str(e)
                }
        
        # Process segments in parallel with ThreadPoolExecutor
        print(f"Starting parallel processing with {num_segments} threads")
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_segments) as executor:
            # Submit all tasks
            futures = {executor.submit(process_segment, seg): seg for seg in segments}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                segment_info = futures[future]
                try:
                    seg_result = future.result()
                    all_results.append(seg_result)
                    print(f"Segment {seg_result['segment_id']} completed with {seg_result['frames_processed']} frames processed")
                except Exception as e:
                    print(f"Error in thread for segment {segment_info[0]}: {e}")
                    all_results.append({
                        "segment_id": segment_info[0], 
                        "frames": {}, 
                        "matches": 0,
                        "start_frame": segment_info[1], 
                        "end_frame": segment_info[2],
                        "error": str(e)
                    })
        
        # Sort results by segment ID to maintain frame order
        all_results.sort(key=lambda x: x["segment_id"])
        
        if pbar:
            pbar.set_description("Writing Final Video")
            
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            return {"success": False, "message": f"Could not create output video: {output_path}"}
            
        # Count total matches and write frames in order
        total_matches = 0
        frames_written = 0
        
        for seg in all_results:
            total_matches += seg["matches"]
            
            # Write each frame for this segment
            for frame_idx in range(seg["start_frame"], seg["end_frame"]):
                if frame_idx in seg["frames"]:
                    out.write(seg["frames"][frame_idx])
                    frames_written += 1
                    
                    if pbar:
                        pbar.set_description(f"Writing frame {frame_idx}/{total_frames}")
                        
        # Clean up resources
        out.release()
        if pbar:
            pbar.close()
            
        print(f"Video processing complete. Output saved to: {output_path}")
        print(f"Found {total_matches} face matches across {total_frames} frames")
        print(f"Wrote {frames_written} frames to output video")
        
        return {
            "success": True, 
            "total_frames": total_frames, 
            "frames_written": frames_written,
            "matches_found": total_matches, 
            "output_path": output_path
        }
                
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error in video processing: {error_msg}")
        if 'pbar' in locals() and pbar:
            pbar.close()
        return {"success": False, "message": str(e), "details": error_msg}
