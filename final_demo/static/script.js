// final_demo/static/script.js

let stream = null;
let capturedImageData = null;

// DOM Elements
const captureSection = document.getElementById('capture-section');
const styleSection = document.getElementById('style-section');
const resultSection = document.getElementById('result-section');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const preview = document.getElementById('preview');
const previewContainer = document.getElementById('preview-container');
const captureBtn = document.getElementById('capture-btn');
const retakeBtn = document.getElementById('retake-btn');
const continueBtn = document.getElementById('continue-btn');
const backBtn = document.getElementById('back-btn');
const resetBtn = document.getElementById('reset-btn');
const styleBtns = document.querySelectorAll('.style-btn');
const loading = document.getElementById('loading');
const resultContainer = document.getElementById('result-container');
const originalImg = document.getElementById('original-img');
const generatedImg = document.getElementById('generated-img');
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

// Generate styled image
async function generateStyled(style) {
    // Reset UI
    loading.style.display = 'block';
    resultContainer.style.display = 'none';
    generatedImg.style.display = 'none';
    errorMessage.style.display = 'none';
    loadingMessage.textContent = 'Starting generation...';
    progressBar.style.width = '0%';
    progressText.textContent = '0%';
    
    // Connect to SSE for progress updates
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource('/progress');
    
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
        // Add timeout to prevent indefinite waiting
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout
        
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ style: style }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const data = await response.json();
        
        if (data.status === 'error') {
            errorMessage.textContent = 'Error: ' + data.message;
            errorMessage.style.display = 'block';
            loading.style.display = 'none';
            resultContainer.style.display = 'block';
        } else {
            // Display results
            originalImg.onload = function() {
                console.log('Original image loaded successfully');
            };
            originalImg.onerror = function() {
                console.error('Failed to load original image');
            };
            originalImg.src = capturedImageData;
            
            generatedImg.onload = function() {
                console.log('Generated image loaded successfully');
                generatedImg.style.display = 'block';
            };
            generatedImg.onerror = function() {
                console.error('Failed to load generated image');
                errorMessage.textContent = 'Error: Could not display generated image';
                errorMessage.style.display = 'block';
            };
            generatedImg.src = data.image;
            
            // Wait a moment for progress bar to show complete, then show results
            setTimeout(() => {
                loading.style.display = 'none';
                resultContainer.style.display = 'block';
            }, 500);
        }
        
        console.log('Generated', style);
    } catch (error) {
        console.error('Generation error:', error);
        let errorMsg = error.message;
        if (error.name === 'AbortError') {
            errorMsg = 'Generation timed out after 5 minutes. Please try again or reduce inference steps.';
        }
        errorMessage.textContent = 'Error: ' + errorMsg;
        errorMessage.style.display = 'block';
        loading.style.display = 'none';
        resultContainer.style.display = 'block';
    } finally {
        if (eventSource) {
            eventSource.close();
        }
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
    if (e.code === 'Space' && captureSection.classList.contains('active') && webcam.style.display !== 'none') {
        e.preventDefault();
        captureImage();
    }
});

retakeBtn.addEventListener('click', () => {
    webcam.style.display = 'block';
    captureBtn.style.display = 'inline';
    previewContainer.style.display = 'none';
});

continueBtn.addEventListener('click', async () => {
    await sendCaptureToServer();
    showSection(styleSection);
});

backBtn.addEventListener('click', () => {
    showSection(captureSection);
});

styleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
        const style = btn.dataset.style;
        showSection(resultSection);
        await generateStyled(style);
    });
});

resetBtn.addEventListener('click', resetApp);

// Initialize
window.addEventListener('load', initWebcam);
