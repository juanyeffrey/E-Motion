// final_demo/static/script.js

let stream = null;
let capturedImageData = null;

// DOM Elements
const captureSection = document.getElementById('capture-section');
const resultSection = document.getElementById('result-section');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('preview-container');
const captureBtn = document.getElementById('capture-btn');
const retakeBtn = document.getElementById('retake-btn');
const generateBtn = document.getElementById('generate-btn');
const resetBtn = document.getElementById('reset-btn');
const loading = document.getElementById('loading');
const resultContainer = document.getElementById('result-container');
const originalImg = document.getElementById('original-img');
const imgAbstract = document.getElementById('img-abstract');
const imgRealistic = document.getElementById('img-realistic');
const imgScifi = document.getElementById('img-scifi');
const loadingMessage = document.getElementById('loading-message');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const errorMessage = document.getElementById('error-message');

let eventSource = null;

// Initialize webcam
async function initWebcam() {
    try {
        console.log('Requesting webcam...');
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
        console.log('Webcam ready');
    } catch (error) {
        console.error('Webcam error:', error);
        alert('Could not access webcam: ' + error.message);
    }
}

// Capture image from webcam
function captureImage() {
    const context = canvas.getContext('2d');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    context.drawImage(webcam, 0, 0);
    capturedImageData = canvas.toDataURL('image/jpeg');
    preview.src = capturedImageData;
    webcam.style.display = 'none';
    captureBtn.style.display = 'none';
    previewContainer.style.display = 'block';
    console.log('Image captured');
}

// Navigate between sections
function showSection(section) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    section.classList.add('active');
}

// Send captured image to server
async function sendCaptureToServer() {
    const response = await fetch('/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: capturedImageData })
    });
    return response.json();
}

// Generate all styled images
async function generateAllStyles() {
    // Switch to result section
    showSection(resultSection);

    // Reset UI
    loading.style.display = 'block';
    resultContainer.style.display = 'none';
    errorMessage.style.display = 'none';
    loadingMessage.textContent = 'Starting generation...';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    
    // Generate unique request ID
    const requestId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
    
    // Connect to SSE for progress updates
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource(`/progress?request_id=${requestId}`);
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.type === 'keepalive') {
            return;
        }
        
        if (data.status === 'generating') {
            loadingMessage.textContent = data.message || 'Generating...';
            progressBar.style.width = data.progress + '%';
            progressText.textContent = data.progress + '%';
        } else if (data.status === 'complete') {
            loadingMessage.textContent = data.message || 'Complete!';
            progressBar.style.width = '100%';
            progressText.textContent = '100%';
            eventSource.close();
        } else if (data.status === 'error') {
            loadingMessage.textContent = 'Error: ' + (data.message || 'Unknown error');
            progressBar.style.background = 'linear-gradient(90deg, #f44336, #d32f2f)';
            eventSource.close();
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('SSE Error:', error);
        eventSource.close();
    };
    
    // Start generation
    try {
        console.log('Starting generation request...');
        // Add timeout to prevent indefinite waiting (increased to 15 mins for CPU usage)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.error('Generation timed out after 15 minutes');
            controller.abort();
        }, 900000); // 15 min timeout
        
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ request_id: requestId }), // Pass request ID
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        console.log('Generation response received');
        
        const data = await response.json();
        
        if (data.status === 'error') {
            errorMessage.textContent = 'Error: ' + data.message;
            errorMessage.style.display = 'block';
            loading.style.display = 'none';
            resultContainer.style.display = 'block';
        } else {
            // Display results
            originalImg.src = capturedImageData;
            
            // Helper to set image source safely
            const setImg = (imgElem, src) => {
                if (src) {
                    console.log(`Setting image for ${imgElem.id}, data length: ${src.length}`);
                    imgElem.src = src;
                    imgElem.style.display = 'block';
                } else {
                    console.error(`Missing image data for ${imgElem.id}`);
                    imgElem.alt = 'Generation failed';
                }
            };

            console.log('Received images:', Object.keys(data.images));
            setImg(imgAbstract, data.images.abstract);
            setImg(imgRealistic, data.images.realistic);
            setImg(imgScifi, data.images.scifi);
            
            // Wait a moment for progress bar to show complete, then show results
            setTimeout(() => {
                loading.style.display = 'none';
                resultContainer.style.display = 'block';
                // Scroll to results
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            }, 500);
        }
    } catch (error) {
        console.error('Generation error:', error);
        errorMessage.textContent = 'Error: ' + error.message;
        errorMessage.style.display = 'block';
        loading.style.display = 'none';
        eventSource.close();
    }
}

// Reset application
async function resetApp() {
    // Close SSE connection
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    await fetch('/reset', { method: 'POST' });
    capturedImageData = null;
    webcam.style.display = 'block';
    captureBtn.style.display = 'inline';
    previewContainer.style.display = 'none';
    resultContainer.style.display = 'none';
    loading.style.display = 'none';
    
    // Reset progress bar
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    progressBar.style.background = 'linear-gradient(90deg, #4CAF50, #45a049)';
    
    showSection(captureSection);
    if (!stream || !stream.active) await initWebcam();
}

// Event Listeners
captureBtn.addEventListener('click', captureImage);

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && webcam.style.display !== 'none') {
        e.preventDefault();
        captureImage();
    }
});

retakeBtn.addEventListener('click', () => {
    webcam.style.display = 'block';
    captureBtn.style.display = 'inline';
    previewContainer.style.display = 'none';
});

generateBtn.addEventListener('click', async () => {
    await sendCaptureToServer();
    await generateAllStyles();
});

resetBtn.addEventListener('click', resetApp);

// Initialize
window.addEventListener('load', initWebcam);
