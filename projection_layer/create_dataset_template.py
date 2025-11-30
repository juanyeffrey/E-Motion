import os
import sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel

# Add project root to path to import perception
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.perception_core import PerceptionPipeline
from diffusion.config import DEVICE

def create_dataset(image_folder, output_file="training_data.npz"):
    """
    Template script to generate training data for the projection layer.
    
    Args:
        image_folder: Directory containing images named like "emotion_description.jpg" 
                      (e.g., "happy_person.jpg", "angry_face.jpg")
        output_file: Path to save the .npz file
    """
    
    # 1. Initialize Pipelines
    print("Initializing Perception Pipeline...")
    perception_pipe = PerceptionPipeline()
    
    print("Initializing CLIP Text Encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    
    perception_vectors = []
    clip_embeddings = []
    
    # 2. Process Data
    # This is a placeholder loop. You should adapt it to your specific dataset structure.
    # For example, iterating over a CSV file with image paths and captions.
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Found {len(image_files)} images. Processing...")
    
    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        
        # --- A. Extract Perception Vector ---
        # Read image (OpenCV format for perception pipe)
        import cv2
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # Run perception
        perception_output = perception_pipe.process_frame(frame)
        
        if perception_output is None:
            print(f"Skipping {img_name}: No face detected.")
            continue
            
        # Get 1441D vector
        p_vec = perception_output.as_vector()
        
        # --- B. Extract CLIP Text Embedding ---
        # Derive prompt from filename or load from a label file
        # Example: "happy_face.jpg" -> "happy face"
        prompt = os.path.splitext(img_name)[0].replace("_", " ")
        
        # Tokenize and encode
        inputs = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE)
        
        with torch.no_grad():
            # Get the sequence output (last_hidden_state) which is [1, 77, 768]
            # We usually want the pooled output or the full sequence depending on how we train.
            # For the projection layer matching SD text encoder, we want the sequence: [1, 77, 768]
            # BUT, our simple MLP projects to [768] (pooled) or we need to project to [77, 768].
            #
            # CURRENT ARCHITECTURE DECISION:
            # The current TextConditioner adds the projection to the text embeddings.
            # Text embeddings are [Batch, Seq, Dim].
            # The projection output is [Batch, Dim] (expanded to [Batch, Seq, Dim]).
            # So we should train against a single vector representation of the text.
            #
            # We can use the pooled_output from CLIP, or the embedding of the EOT token.
            # Let's use the pooled_output for semantic stability.
            
            text_outputs = text_encoder(input_ids)
            c_embed = text_outputs.pooler_output  # [1, 768]
            
        # Store
        perception_vectors.append(p_vec)
        clip_embeddings.append(c_embed.cpu().numpy().flatten())
        
        if len(perception_vectors) % 10 == 0:
            print(f"Processed {len(perception_vectors)} samples...")

    # 3. Save Dataset
    if not perception_vectors:
        print("No valid data found.")
        return

    perception_arr = np.array(perception_vectors, dtype=np.float32)
    clip_arr = np.array(clip_embeddings, dtype=np.float32)
    
    print(f"Saving dataset to {output_file}...")
    print(f"Perception Shape: {perception_arr.shape}")
    print(f"CLIP Shape: {clip_arr.shape}")
    
    np.savez(output_file, perception_embeddings=perception_arr, clip_embeddings=clip_arr)
    print("Done!")

if __name__ == "__main__":
    # Example usage:
    # python projection_layer/create_dataset_template.py --images ./my_dataset_folder
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="Path to folder containing images")
    parser.add_argument("--out", default="training_data.npz", help="Output .npz file")
    args = parser.parse_args()
    
    if args.images:
        create_dataset(args.images, args.out)
    else:
        print("Please provide an image folder using --images")
