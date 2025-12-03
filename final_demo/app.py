# final_demo/app.py

from flask import Flask, render_template, request, jsonify, send_file, Response
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import sys
import queue
import json
import time
from threading import Thread

# Add parent directory to path to import perception and diffusion modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Store the captured image temporarily
captured_image = None
generated_image = None

# Global pipelines (initialized on first use for faster startup)
perception_pipeline = None
diffusion_pipeline = None

# Progress tracking for loading bar
progress_queue = queue.Queue()

def initialize_pipelines():
    """Lazy initialization of heavy pipelines"""
    global perception_pipeline, diffusion_pipeline
    
    if perception_pipeline is None:
        print("[INIT] Loading perception pipeline...")
        from perception.perception_core import PerceptionPipeline
        perception_pipeline = PerceptionPipeline()
        print("[INIT] Perception pipeline loaded!")
    
    if diffusion_pipeline is None:
        print("[INIT] Loading diffusion pipeline (this may take 1-2 minutes on first run)...")
        from diffusion.diffusion import StyleTransferDiffusion
        from diffusion.config import PERCEPTION_DIM_NO_AUDIO
        
        # Use the configured perception dimension
        perception_dim = PERCEPTION_DIM_NO_AUDIO
        
        print(f"[INIT] Using perception_dim={perception_dim}")
        diffusion_pipeline = StyleTransferDiffusion(perception_dim=perception_dim)
        print("[INIT] Diffusion pipeline loaded!")

@app.route('/')
def index():
    """Main page with camera interface"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Simple webcam test page"""
    return render_template('test.html')

@app.route('/capture', methods=['POST'])
def capture_image():
    """Receive and store the captured image from webcam"""
    global captured_image
    
    try:
        data = request.json
        print(f"[CAPTURE] Received request with keys: {data.keys() if data else 'None'}")
        
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400
        
        image_data = data['image']
        print(f"[CAPTURE] Image data length: {len(image_data)}")
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        image_bytes = base64.b64decode(image_data)
        print(f"[CAPTURE] Decoded bytes length: {len(image_bytes)}")
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        captured_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if captured_image is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400
        
        print(f"[CAPTURE] Image decoded successfully: {captured_image.shape}")
        return jsonify({'status': 'success', 'message': 'Image captured successfully'})
        
    except Exception as e:
        print(f"[CAPTURE ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/generate', methods=['POST'])
def generate_styled_image():
    """Generate all 3 styles: Abstract, Realistic, Combo"""
    global captured_image, generated_image
    
    try:
        # Initialize pipelines on first use
        initialize_pipelines()
        
        # data = request.json
        # style_choice = data['style']  # Ignored, we generate all
        
        print(f"[GENERATE] Generating all styles...")
        
        if captured_image is None:
            print("[GENERATE ERROR] No captured image available")
            return jsonify({'status': 'error', 'message': 'No image captured'}), 400
        
        print(f"[GENERATE] Processing image of shape: {captured_image.shape}")
        
        # Step 1: Extract perception embeddings
        print("[GENERATE] Extracting perception embeddings...")
        perception_output = perception_pipeline.process_image(captured_image)
        perception_vec = perception_output.as_vector()
        
        if perception_vec is None:
            print("[GENERATE ERROR] No face detected in image")
            return jsonify({
                'status': 'error', 
                'message': 'No face detected. Please ensure your face is clearly visible and try again.'
            }), 400
        
        print(f"[GENERATE] Perception vector shape: {perception_vec.shape}")
        emotion = perception_output.get_dominant_emotion()
        print(f"[GENERATE] Detected emotion: {emotion}")
        
        # Step 2: Generate all 3 styles
        styles = ['abstract', 'realistic', 'scifi']
        results = {}
        
        total_styles = len(styles)
        
        for i, style_choice in enumerate(styles):
            print(f"[GENERATE] Generating {style_choice} ({i+1}/{total_styles})...")
            
            # Send progress update
            progress_queue.put({
                'status': 'generating', 
                'progress': int((i / total_styles) * 100), 
                'message': f'Generating {style_choice} ({i+1}/{total_styles})...'
            })
            
            # Generate with progress callback (mapped to sub-progress)
            def progress_callback(step, total_steps):
                # Calculate global progress: base for this style + fraction of this style
                style_base = (i / total_styles) * 100
                style_fraction = (step / total_steps) * (100 / total_styles)
                global_progress = int(style_base + style_fraction)
                
                progress_queue.put({
                    'status': 'generating',
                    'progress': global_progress,
                    'step': step,
                    'total_steps': total_steps,
                    'message': f'Generating {style_choice}... ({step}/{total_steps})'
                })
            
            generated_pil = diffusion_pipeline.generate(
                user_image=captured_image,
                perception_vec=perception_vec,
                style_choice=style_choice,
                progress_callback=progress_callback,
                emotion_label=emotion
            )
            
            # Convert to base64
            buffered = io.BytesIO()
            generated_pil.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            results[style_choice] = f'data:image/jpeg;base64,{img_base64}'
        
        progress_queue.put({'status': 'complete', 'progress': 100, 'message': 'All generations complete!'})
        print("[GENERATE] All images generated successfully!")
        
        return jsonify({
            'status': 'success',
            'images': results,
            'message': 'Generated all styles',
            'emotion': emotion
        })
        
    except Exception as e:
        print(f"[GENERATE ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

        print(f"[GENERATE ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/progress')
def progress():
    """Server-Sent Events endpoint for progress updates"""
    def generate():
        while True:
            try:
                # Get progress update from queue (with timeout)
                message = progress_queue.get(timeout=30)
                yield f"data: {json.dumps(message)}\n\n"
                
                # Stop if generation is complete or failed
                if message.get('status') in ['complete', 'error']:
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {{\"type\": \"keepalive\"}}\\n\\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the application state"""
    global captured_image, generated_image
    captured_image = None
    generated_image = None
    
    # Clear progress queue
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break
    
    return jsonify({'status': 'success', 'message': 'Reset successful'})

@app.route('/reference/<style>')
def get_reference(style):
    """Serve reference images"""
    reference_path = os.path.join('static', 'references', f'{style}.jpg')
    if os.path.exists(reference_path):
        return send_file(reference_path, mimetype='image/jpeg')
    else:
        # Return placeholder if reference doesn't exist yet
        return '', 404

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/references', exist_ok=True)
    
    print("="*60)
    print("E-Motion Style Transfer Demo")
    print("="*60)
    print("Architecture:")
    print("  • Abstract style: Text-only (NO IP-Adapter, 3 steps)")
    print("  • Realistic style: ControlNet + IP-Adapter (4 steps)")
    print("  • Scifi style: ControlNet + IP-Adapter (4 steps)")
    print("="*60)
    print("Open http://localhost:5000 in your browser")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
