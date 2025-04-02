input_image_path = "/content/input1.png"
target_image_path = "/content/00037.jpg"
input_video_path = "/content/drive/MyDrive/Projects/input_crowded.mp4"
output_video_path = "/content/drive/MyDrive/Projects/output.mp4"
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from scipy.spatial.distance import cosine


# Function to load and preprocess image
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Function to detect faces using MTCNN
def detect_faces(image):
    detector = MTCNN()
    return detector.detect_faces(image)

# Function to extract face from image
def extract_face(image, face_info, required_size=(160, 160)):
    x, y, width, height = face_info['box']
    # Ensure coordinates are not negative
    x, y = max(0, x), max(0, y)
    face = image[y:y+height, x:x+width]
    face_image = cv2.resize(face, required_size)
    face_image = face_image.astype('float32') / 255.0  # Normalize
    return face_image, (x, y, width, height)

# Updated function for PyTorch model compatibility
def get_embedding(face_image, model):
    """Get face embedding using PyTorch VGGFace2 model"""
    # Ensure face_image is in RGB format (PyTorch models expect RGB)
    if face_image.shape[2] == 4:  # RGBA
        face_image = face_image[:, :, :3]
    
    # PyTorch models expect different preprocessing than TensorFlow
    # Normalize in range [-1, 1] or according to ImageNet stats
    face_image = (face_image * 2) - 1  # Scale from [0,1] to [-1,1]
    
    # Get embedding using the predict wrapper we added
    embedding = model.predict(face_image)[0]
    
    # Normalize embedding to have unit length
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


# Function for face alignment
def align_face(img, landmarks):
    """Align face based on eye positions"""
    if landmarks is None:
        return img
    
    # Extract eye landmarks
    left_eye = np.array([landmarks['left_eye']])
    right_eye = np.array([landmarks['right_eye']])
    
    # Calculate angle
    dY = right_eye[0, 1] - left_eye[0, 1]
    dX = right_eye[0, 0] - left_eye[0, 0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Get the center of the image
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    # Rotate the image to align the eyes horizontally
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return aligned

# Function to enhance face recognition with preprocessing
def enhanced_find_person(reference_path, target_path, similarity_threshold=0.5):
    """Enhanced version with face alignment and better preprocessing"""
    try:
        # Load the FaceNet model
        model = load_facenet_model()
        
        # Load images
        reference_image = load_image(reference_path)
        target_image = load_image(target_path)
        
        # Detect faces
        reference_faces = detect_faces(reference_image)
        target_faces = detect_faces(target_image)
        
        if not reference_faces:
            return {"success": False, "message": "No faces detected in the reference image"}
        
        if not target_faces:
            return {"success": False, "message": "No faces detected in the target image"}
        
        # Get the first face from reference image (with alignment)
        ref_face_info = reference_faces[0]
        ref_face, ref_box = extract_face(reference_image, ref_face_info)
        
        # Align reference face using landmarks
        if 'keypoints' in ref_face_info:
            ref_face = align_face(ref_face, ref_face_info['keypoints'])
            
        ref_embedding = get_embedding(ref_face, model)
        
        # Look for matching faces in the target image
        matches = []
        aligned_faces = []
        
        for face_info in target_faces:
            face, box = extract_face(target_image, face_info)
            
            # Align target face
            if 'keypoints' in face_info:
                face = align_face(face, face_info['keypoints'])
            
            aligned_faces.append((face, box))
            embedding = get_embedding(face, model)
            
            # Calculate similarity (1 - cosine distance)
            similarity = 1 - cosine(ref_embedding, embedding)
            
            if similarity > similarity_threshold:
                matches.append({
                    "box": box,
                    "similarity": similarity,
                    "face": face
                })
                
        # Visualize aligned faces
        plt.figure(figsize=(15, 6))
        plt.subplot(1, len(aligned_faces) + 1, 1)
        plt.imshow(normalize_for_display(ref_face))
        plt.title("Reference (Aligned)")
        plt.axis('off')

        for i, (face, _) in enumerate(aligned_faces):
            plt.subplot(1, len(aligned_faces) + 1, i + 2)
            plt.imshow(normalize_for_display(face))
            plt.title(f"Face {i+1}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
        
        # Create a visualization of results (same as before)
        output_image = target_image.copy()
        
        # Sort matches by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Visual output and match highlighting
        plt.figure(figsize=(15, 15))
        plt.imshow(normalize_for_display(target_image))
        
        # Draw rectangles for all faces
        for face_info in target_faces:
            x, y, width, height = face_info['box']
            plt.gca().add_patch(plt.Rectangle((x, y), width, height, 
                                             fill=False, color='blue', linewidth=2))
            
        # Highlight matches with green color and similarity score
        for match in matches:
            x, y, width, height = match['box']
            similarity = match['similarity']
            plt.gca().add_patch(plt.Rectangle((x, y), width, height, 
                                             fill=False, color='green', linewidth=3))
            plt.text(x, y-10, f"{similarity:.2f}", 
                    backgroundcolor='green', color='white', fontsize=12)
            
        plt.title("Target Image - Blue: All Faces, Green: Matches")
        plt.axis('off')
        plt.show()
        
        # Create a summary of results
        result = {
            "success": True,
            "reference_faces": len(reference_faces),
            "target_faces": len(target_faces),
            "matches": len(matches),
            "matches_details": matches
        }
        
        return result
    
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}
# Fixed function to load face recognition model using facenet-pytorch
def load_facenet_model():
    """Load VGGFace2 model from facenet-pytorch library"""
    print("Loading VGGFace2 model from facenet-pytorch...")
    try:
        # Check if facenet-pytorch is installed, if not install it
        try:
            import facenet_pytorch
        except ImportError:
            print("Installing facenet-pytorch...")
            import subprocess
            subprocess.check_call(["pip", "install", "facenet-pytorch"])
            import facenet_pytorch
        
        # Import InceptionResnetV1 from facenet-pytorch
        from facenet_pytorch import InceptionResnetV1
        
        # Load the VGGFace2 pretrained model
        model = InceptionResnetV1(pretrained='vggface2').eval()
        print("VGGFace2 model loaded successfully from facenet-pytorch.")
        
        # If running in Colab with CUDA
        import torch
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Model moved to GPU.")
        
        # Create a wrapper function for TensorFlow-like interface compatibility
        def predict_wrapper(img_array):
            # Convert numpy array to PyTorch tensor
            import torch
            if isinstance(img_array, np.ndarray):
                # Add batch dimension if not already
                if len(img_array.shape) == 3:
                    img_array = np.expand_dims(img_array, axis=0)
                
                # Convert to PyTorch tensor
                tensor = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    tensor = tensor.to('cuda')
                
                # Get embedding
                with torch.no_grad():
                    embedding = model(tensor).cpu().numpy()
                
                return embedding
            else:
                raise ValueError("Input must be a numpy array")
        
        # Add predict method to model for compatibility with existing code
        model.predict = predict_wrapper
        
        return model
    
    except Exception as e:
        raise Exception(f"Failed to load VGGFace2 model from facenet-pytorch: {e}")
# Add this helper function to your code
def normalize_for_display(image):
    """Normalize image to [0,1] range for display with matplotlib"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # For floating point images
        return np.clip(image, 0, 1)
    else:
        # For integer images
        return np.clip(image, 0, 255).astype(np.uint8)
# Function to find a person in a crowd
def find_person_in_crowd(reference_path, target_path, similarity_threshold=0.4):
    try:
        # Load the FaceNet model
        model = load_facenet_model()
        
        # Load images
        reference_image = load_image(reference_path)
        target_image = load_image(target_path)
        
        # Detect faces
        reference_faces = detect_faces(reference_image)
        target_faces = detect_faces(target_image)
        
        if not reference_faces:
            return {"success": False, "message": "No faces detected in the reference image"}
        
        if not target_faces:
            return {"success": False, "message": "No faces detected in the target image"}
        
        # Get the first face from reference image (assuming it's the main person)
        ref_face, ref_box = extract_face(reference_image, reference_faces[0])
        ref_embedding = get_embedding(ref_face, model)
        
        # Look for matching faces in the target image
        matches = []
        for face_info in target_faces:
            face, box = extract_face(target_image, face_info)
            embedding = get_embedding(face, model)
            
            # Calculate similarity (1 - cosine distance)
            similarity = 1 - cosine(ref_embedding, embedding)
            
            if similarity > similarity_threshold:
                matches.append({
                    "box": box,
                    "similarity": similarity
                })
        
        # Create a visualization of results
        output_image = target_image.copy()
        
        # Sort matches by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Draw reference face
        ref_x, ref_y, ref_w, ref_h = ref_box
        plt.figure(figsize=(5, 5))
        ref_face_display = normalize_for_display(reference_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w])
        plt.imshow(ref_face_display)
        plt.title("Reference Face")
        plt.axis('off')
        plt.show()
        
        # Draw all faces in target image
        plt.figure(figsize=(15, 15))
        plt.imshow(normalize_for_display(target_image))
        
        # Draw rectangles for all faces
        for face_info in target_faces:
            x, y, width, height = face_info['box']
            plt.gca().add_patch(plt.Rectangle((x, y), width, height, 
                                             fill=False, color='blue', linewidth=2))
            
        # Highlight matches with green color and similarity score
        for match in matches:
            x, y, width, height = match['box']
            similarity = match['similarity']
            plt.gca().add_patch(plt.Rectangle((x, y), width, height, 
                                             fill=False, color='green', linewidth=3))
            plt.text(x, y-10, f"{similarity:.2f}", 
                    backgroundcolor='green', color='white', fontsize=12)
            
        plt.title("Target Image - Blue: All Faces, Green: Matches")
        plt.axis('off')
        plt.show()
        
        # Create a summary of results
        result = {
            "success": True,
            "reference_faces": len(reference_faces),
            "target_faces": len(target_faces),
            "matches": len(matches),
            "matches_details": matches
        }
        
        return result
    
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}



# Main function to run the face recognition system
def main():
    import os
    
    print("Welcome to Enhanced Face Recognition System")
    print("------------------------------------------")
    
    reference_path = input_image_path
    target_path = target_image_path
    
    if not os.path.exists(reference_path) or not os.path.exists(target_path):
        print("One or both of the specified files do not exist.")
        return
    
    print("\nProcessing images...")
    
    # Use the enhanced version with face alignment
    result = enhanced_find_person(reference_path, target_path, similarity_threshold=0.4)
    
    if not result["success"]:
        print(f"Error: {result['message']}")
        return
    
    print("\nResults Summary:")
    print(f"- Found {result['reference_faces']} faces in the reference image")
    print(f"- Found {result['target_faces']} faces in the target image")
    print(f"- Found {result['matches']} potential matches")
    
    if result['matches'] > 0:
        print("\nTop match similarity:", result['matches_details'][0]['similarity'])
    else:
        print("\nNo matches found. Try adjusting the similarity threshold.")
        

# Run the main function when script is executed: Image + Image
if __name__ == "__main__":
    main()
    
#  Video Section
def optimized_process_video(video_path, reference_path, output_path, 
                           similarity_threshold=0.4, frame_skip=2, 
                           display_progress=True, resize_factor=0.5,
                           num_segments=4):
    """
    Optimized video processing that divides the video into segments and processes 
    them in parallel for maximum performance.
    
    Args:
        video_path: Path to input video
        reference_path: Path to reference face image
        output_path: Path for output video
        similarity_threshold: Threshold for face matching (0.0-1.0)
        frame_skip: Process every nth frame
        display_progress: Whether to display a progress bar
        resize_factor: Factor to resize frames for faster processing
        num_segments: Number of segments to divide the video into
    """
    import concurrent.futures
    import threading
    import time
    import numpy as np
    from collections import defaultdict
    
    if display_progress:
        from tqdm.notebook import tqdm
        
    # Thread-safe counter for progress updates
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
    
    # Process a segment of video
    def process_segment(segment_info):
        segment_id, start_frame, end_frame, segment_path = segment_info
        
        # Create a video capture for this segment
        cap = cv2.VideoCapture(video_path)
        
        # Set the starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize variables
        frame_idx = start_frame
        segment_frames = {}
        segment_matches = 0
        
        # Process frames in this segment
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Update reading progress
            progress.update(1, f"Reading [{segment_id}]")
            
            # Process only selected frames based on frame_skip
            if (frame_idx % frame_skip == 0):
                # Resize frame for faster processing
                resized = cv2.resize(frame, (process_width, process_height))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = detector.detect_faces(rgb)
                frame_matches = 0
                
                # Process faces
                for face in faces:
                    x, y, w, h = face['box']
                    # Scale coordinates back to original size
                    x_orig = int(x / resize_factor)
                    y_orig = int(y / resize_factor)
                    w_orig = int(w / resize_factor)
                    h_orig = int(h / resize_factor)
                    
                    # Ensure coordinates are valid
                    x_orig = max(0, x_orig)
                    y_orig = max(0, y_orig)
                    w_orig = min(w_orig, frame.shape[1] - x_orig)
                    h_orig = min(h_orig, frame.shape[0] - y_orig)
                    
                    if w_orig <= 0 or h_orig <= 0:
                        continue
                        
                    # Draw blue rectangle for all faces
                    cv2.rectangle(frame, (x_orig, y_orig), 
                                 (x_orig + w_orig, y_orig + h_orig), 
                                 (255, 0, 0), 2)
                    
                    # Extract face for recognition
                    face_img = frame[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
                    if face_img.size == 0:
                        continue
                        
                    # Process face for recognition
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))
                    face_normalized = face_resized.astype('float32') / 255.0
                    
                    # Get embedding and calculate similarity
                    embedding = get_embedding(face_normalized, model)
                    similarity = 1 - cosine(ref_embedding, embedding)
                    
                    # Highlight matches
                    if similarity > similarity_threshold:
                        cv2.rectangle(frame, (x_orig, y_orig), 
                                     (x_orig + w_orig, y_orig + h_orig), 
                                     (0, 255, 0), 3)
                        cv2.putText(frame, f"{similarity:.2f}", 
                                   (x_orig, y_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.9, (0, 255, 0), 2)
                        frame_matches += 1
                        segment_matches += 1
                
                # Add frame number
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Update processing progress
                progress.update(1, f"Processing [{segment_id}]")
            
            # Store the frame (original or processed)
            segment_frames[frame_idx] = frame
            
            # Move to next frame
            frame_idx += 1
            
        # Release segment capture
        cap.release()
        
        return {
            "segment_id": segment_id,
            "frames": segment_frames,
            "matches": segment_matches,
            "start_frame": start_frame,
            "end_frame": end_frame
        }
    
    try:
        # Load model and reference face once for all segments
        model = load_facenet_model()
        reference_image = load_image(reference_path)
        reference_faces = detect_faces(reference_image)
        
        if not reference_faces:
            return {"success": False, "message": "No face detected in reference image"}
            
        ref_face_info = reference_faces[0]
        ref_face, ref_box = extract_face(reference_image, ref_face_info)
        
        # Align reference face if landmarks are available
        if 'keypoints' in ref_face_info:
            ref_face = align_face(ref_face, ref_face_info['keypoints'])
            
        # Get reference face embedding
        ref_embedding = get_embedding(ref_face, model)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Calculate processing dimensions
        process_width = int(width * resize_factor)
        process_height = int(height * resize_factor)
        
        # Set up face detector
        detector = MTCNN()
        
        # Divide video into segments
        frames_per_segment = total_frames // num_segments
        segments = []
        
        for i in range(num_segments):
            start_frame = i * frames_per_segment
            end_frame = (i + 1) * frames_per_segment if i < num_segments - 1 else total_frames
            segment_path = f"temp_segment_{i}.mp4"
            segments.append((i, start_frame, end_frame, segment_path))
            
        # Create progress bar with phases
        total_ops = total_frames * 2  # Reading + Processing
        if display_progress:
            pbar = tqdm(total=total_ops, desc="Initializing...")
        else:
            pbar = None
            
        # Create progress counter
        progress = ProgressCounter(total_ops, pbar)
        
        # Process segments in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_segments) as executor:
            # Submit all segment processing tasks
            future_to_segment = {
                executor.submit(process_segment, segment): segment
                for segment in segments
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    print(f'Segment {segment[0]} generated an exception: {exc}')
                    raise
                    
        # Sort results by segment ID to ensure correct order
        all_results.sort(key=lambda x: x["segment_id"])
        
        # Merge results and write final video
        if display_progress:
            pbar.set_description("Writing Final Video")
            
        # Create video writer for the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write all frames in order
        total_matches = 0
        for segment_result in all_results:
            segment_frames = segment_result["frames"]
            total_matches += segment_result["matches"]
            
            # Write frames in this segment in order
            for frame_idx in range(segment_result["start_frame"], segment_result["end_frame"]):
                if frame_idx in segment_frames:
                    out.write(segment_frames[frame_idx])
                    
                    # Update writing progress (no need to count in total_ops)
                    if display_progress:
                        pbar.set_description(f"Writing {frame_idx}/{total_frames}")
                        
        # Clean up
        out.release()
        if display_progress:
            pbar.close()
            
        return {
            "success": True,
            "total_frames": total_frames,
            "processed_frames": total_frames,
            "matches_found": total_matches,
            "output_path": output_path
        }
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        if display_progress and 'pbar' in locals():
            pbar.close()
        return {"success": False, "message": str(e), "details": error_msg}
# Process video with optimized function
result = optimized_process_video(
    video_path=input_video_path,
    reference_path=input_image_path, 
    output_path=output_video_path,
    similarity_threshold=0.4,
    frame_skip=1,
    resize_factor=0.5,
    num_segments=4  
)

if result["success"]:
    print(f"Video processed successfully: {result['matches_found']} matches found")
else:
    print(f"Processing failed: {result['message']}")

