import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Deepfake Detection App", layout="wide")
st.title("Deepfake Detection App")

@st.cache_resource
def load_model():
    # Load model
    model_type = 'inception_next_tiny'
    model = timm.create_model(model_type, pretrained=True)
    
    # Check if model file exists
    model_path = st.session_state.get('model_path', None)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        return None

# Define image transformation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names
class_names = ['fake', 'real']

# Sidebar for model upload
with st.sidebar:
    st.header("Model Settings")
    model_file = st.file_uploader("Upload model file (.pt)", type=["pt"])
    
    if model_file:
        # Save the uploaded model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(model_file.getvalue())
            st.session_state['model_path'] = tmp_file.name
        st.success("Model uploaded successfully!")
    
    st.header("Processing Settings")
    frame_interval = st.slider("Process every Nth frame", 1, 30, 10)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

# Main content
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name
    
    st.video(video_path)
    
    # Process button
    if st.button("Detect Deepfake"):
        model = load_model()
        
        if model is None:
            st.error("Please upload a model file first!")
        else:
            # Process the video
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            frame_count = 0
            processed_count = 0
            sample_frames = []
            predictions = []
            confidence_scores = []
            
            col1, col2 = st.columns(2)
            results_container = st.container()
            
            with results_container:
                st.subheader("Processing Results")
                results_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Update progress
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
                
                if frame_count % frame_interval != 0:
                    continue
                
                processed_count += 1
                
                # Convert frame to PIL image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_image = test_transform(frame_pil).unsqueeze(0)
                
                # Perform inference
                with torch.no_grad():
                    output = model(input_image)
                
                # Get predictions
                probabilities = F.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities, dim=0).item()
                confidence = probabilities[predicted_class].item()
                predicted_class_name = class_names[predicted_class]
                
                predictions.append(predicted_class_name)
                confidence_scores.append(confidence)
                
                # Annotate frame
                label = f'{predicted_class_name} ({confidence:.2f})'
                color = (0, 255, 0) if predicted_class_name == 'real' else (255, 0, 0)
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Store sample frames (evenly distributed)
                if len(sample_frames) < 6 and processed_count % max(1, processed_count // 6) == 0:
                    sample_frames.append((annotated_frame, predicted_class_name, confidence))
            
            cap.release()
            
            # Display final results
            if predictions:
                fake_count = predictions.count('fake')
                real_count = predictions.count('real')
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                
                final_result = 'fake' if fake_count > real_count else 'real'
                confidence_percentage = (max(fake_count, real_count) / len(predictions)) * 100
                
                results_placeholder.markdown(f"""
                ### Results Summary
                - **Final Classification: {final_result.upper()}**
                - Confidence: {confidence_percentage:.1f}%
                - Average confidence score: {avg_confidence:.2f}
                - Frames classified as fake: {fake_count} ({fake_count/len(predictions)*100:.1f}%)
                - Frames classified as real: {real_count} ({real_count/len(predictions)*100:.1f}%)
                - Total frames processed: {len(predictions)}
                """)
                
                # Display sample frames in a grid
                st.subheader("Sample Frames")
                cols = st.columns(3)
                
                for i, (frame, label, conf) in enumerate(sample_frames):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption=f"Frame: {label} (Conf: {conf:.2f})",
                                use_container_width=True)  # Updated parameter here
                
                # Create a simple chart
                st.subheader("Classification Distribution")
                st.bar_chart({
                    'Fake': [fake_count],
                    'Real': [real_count]
                })
            
            # Clean up temporary file
            os.unlink(video_path)
            progress_bar.empty()
            status_text.empty()

st.markdown("""
### How to use this app:
1. Upload your trained model file (.pt) in the sidebar
2. Upload a video file to analyze
3. Click "Detect Deepfake" to process the video
4. View the results and sample frames

The app will classify each frame as either "fake" or "real" and provide an overall classification for the video.
""")