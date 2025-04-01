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

# Function to find a person in a crowd
def find_person_in_crowd(reference_path, target_path, similarity_threshold=0.5):
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
        plt.imshow(reference_image[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w])
        plt.title("Reference Face")
        plt.axis('off')
        plt.show()
        
        # Draw all faces in target image
        plt.figure(figsize=(15, 15))
        plt.imshow(target_image)
        
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
        plt.imshow(ref_face)
        plt.title("Reference (Aligned)")
        plt.axis('off')
        
        for i, (face, _) in enumerate(aligned_faces):
            plt.subplot(1, len(aligned_faces) + 1, i + 2)
            plt.imshow(face)
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
        plt.imshow(target_image)
        
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
        
    # For Video Processing
    def process_video(video_path, reference_path, output_path, similarity_threshold=0.4, fps=None, frame_skip=1):
        """
        Process a video by finding a person in each frame and highlighting matches.
        
        Args:
            video_path (str): Path to the input video file
            reference_path (str): Path to the reference face image
            output_path (str): Path for the output video
            similarity_threshold (float): Threshold for face matching (0.0-1.0)
            fps (float): Frames per second for output video (None uses input video's fps)
            frame_skip (int): Process every nth frame to speed up processing
            
        Returns:
            dict: Summary of processing results
        """
        print(f"Processing video: {video_path}")
        print(f"Reference face: {reference_path}")
        
        # Check if input files exist
        if not os.path.exists(video_path):
            return {"success": False, "message": f"Video file not found: {video_path}"}
        if not os.path.exists(reference_path):
            return {"success": False, "message": f"Reference image not found: {reference_path}"}
        
        try:
            # Load the face recognition model once
            model = load_facenet_model()
            print("Face recognition model loaded successfully.")
            
            # Load reference face
            reference_image = load_image(reference_path)
            reference_faces = detect_faces(reference_image)
            
            if not reference_faces:
                return {"success": False, "message": "No face detected in the reference image"}
            
            # Process reference face
            ref_face_info = reference_faces[0]
            ref_face, ref_box = extract_face(reference_image, ref_face_info)
            
            # Align reference face if landmarks are available
            if 'keypoints' in ref_face_info:
                ref_face = align_face(ref_face, ref_face_info['keypoints'])
                
            # Get reference face embedding
            ref_embedding = get_embedding(ref_face, model)
            print("Reference face processed successfully.")
            
            # Open the video
            video = cv2.VideoCapture(video_path)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get video FPS
            if fps is None:
                fps = video.get(cv2.CAP_PROP_FPS)
            
            # Total frames
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can also use 'XVID'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            frame_count = 0
            processed_count = 0
            matches_found = 0
            
            print(f"Starting video processing: {total_frames} total frames")
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                    
                # Skip frames to speed up processing
                if frame_count % frame_skip != 0:
                    # For skipped frames, just write original frame
                    out.write(frame)
                    frame_count += 1
                    continue
                
                # Convert frame from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the frame
                faces = detect_faces(rgb_frame)
                
                # Process each face in the frame
                frame_matches = 0
                for face_info in faces:
                    face, box = extract_face(rgb_frame, face_info)
                    
                    # Align face if landmarks are available
                    if 'keypoints' in face_info:
                        aligned_face = align_face(face, face_info['keypoints'])
                    else:
                        aligned_face = face
                    
                    # Get face embedding
                    embedding = get_embedding(aligned_face, model)
                    
                    # Calculate similarity
                    similarity = 1 - cosine(ref_embedding, embedding)
                    
                    # Draw all detected faces with blue rectangles
                    x, y, width, height = face_info['box']
                    cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
                    
                    # If similarity is above threshold, highlight the match
                    if similarity > similarity_threshold:
                        # Draw green rectangle for matches
                        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 3)
                        
                        # Display similarity score
                        text = f"{similarity:.2f}"
                        cv2.putText(frame, text, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        frame_matches += 1
                        matches_found += 1
                
                # Add frame number
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Write frame to output video
                out.write(frame)
                
                # Update counters
                frame_count += 1
                processed_count += 1
                
                # Print progress
                if processed_count % 10 == 0 or processed_count == 1:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%), Found {matches_found} matches so far")
            
            # Release video resources
            video.release()
            out.release()
            
            print(f"Video processing complete. Output saved to {output_path}")
            print(f"Processed {processed_count} of {frame_count} frames, found {matches_found} matches")
            
            return {
                "success": True,
                "total_frames": frame_count,
                "processed_frames": processed_count,
                "matches_found": matches_found,
                "output_path": output_path
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "message": f"Error processing video: {str(e)}"}
        
    # Calling Video Processing
    # Example usage
    result = process_video(
        video_path=input_video_path,
        reference_path=input_image_path,
        output_path=output_video_path,
        similarity_threshold=0.4,  # Adjust based on your needs
        frame_skip=1  # Process every n frame for faster processing
    )

    if result["success"]:
        print(f"Video processed successfully. Found {result['matches_found']} matches.")
    else:
        print(f"Video processing failed: {result['message']}")