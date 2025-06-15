function renderFaceImageTables(data) {
    const emotions = data.facial_emotions || [];

    if (emotions.length === 0) {
        document.getElementById("faceEmotionTableContainer").innerHTML = "<p>Aucune émotion détectée.</p>";
        document.getElementById("faceSentimentTableContainer").innerHTML = "<p>Aucun sentiment détecté.</p>";
        return;
    }

    // Collecte toutes les clés uniques d'émotions et sentiments pour créer les colonnes
    const allEmotionLabels = new Set();
    const allSentimentLabels = new Set();

    emotions.forEach(face => {
        Object.keys(face.emotions || {}).forEach(e => allEmotionLabels.add(e));
        Object.keys(face.sentiments || {}).forEach(s => allSentimentLabels.add(s));
    });

    const emotionLabels = Array.from(allEmotionLabels);
    const sentimentLabels = Array.from(allSentimentLabels);

    // --- Tableau des émotions ---
    let emotionTableHTML = `
        <table class="result-table">
            <thead>
                <tr>
                    <th>Face #</th>
                    ${emotionLabels.map(e => `<th>${capitalize(e)}</th>`).join('')}
                </tr>
            </thead>
            <tbody>`;

    emotions.forEach((face, index) => {
        emotionTableHTML += `<tr><td>${index + 1}</td>`;

        emotionLabels.forEach(label => {
            const score = face.emotions && face.emotions[label] ? face.emotions[label] : 0;
            emotionTableHTML += `<td>${(score * 100).toFixed(1)}</td>`;
        });

        emotionTableHTML += `</tr>`;
    });

    emotionTableHTML += `</tbody></table>`;

    // --- Tableau des sentiments ---
    let sentimentTableHTML = `
        <table class="result-table">
            <thead>
                <tr>
                    <th>Face #</th>
                    ${sentimentLabels.map(s => `<th>${capitalize(s)}</th>`).join('')}
                </tr>
            </thead>
            <tbody>`;

    emotions.forEach((face, index) => {
        sentimentTableHTML += `<tr><td>${index + 1}</td>`;

        sentimentLabels.forEach(label => {
            const score = face.sentiments && face.sentiments[label] ? face.sentiments[label] : 0;
            sentimentTableHTML += `<td>${(score * 100).toFixed(1)}</td>`;
        });

        sentimentTableHTML += `</tr>`;
    });

    sentimentTableHTML += `</tbody></table>`;

    // Injection dans le DOM
    document.getElementById("faceEmotionTableContainer").innerHTML = emotionTableHTML;
    document.getElementById("faceSentimentTableContainer").innerHTML = sentimentTableHTML;
}

function displayVideoEmotionSentimentTables(data) {
    const container = document.getElementById('videoEmotionSentimentTableContainer');
    container.innerHTML = ''; // reset

    const emotionsSet = new Set();
    const sentimentsSet = new Set();

    // Récupérer toutes les émotions et sentiments uniques présentes dans les frames
    data.emotions_timeline.forEach(frame => {
        frame.detected_faces.forEach(face => {
            Object.keys(face.emotions).forEach(e => emotionsSet.add(e));
            Object.keys(face.sentiments).forEach(s => sentimentsSet.add(s));
        });
    });

    const emotionsList = Array.from(emotionsSet).sort();
    const sentimentsList = Array.from(sentimentsSet).sort();

    // --- Construction du tableau émotions ---
    const emotionsTable = document.createElement('table');
    emotionsTable.classList.add('result-table');

    // Header
    let headerRow = '<tr><th>Temps (s)</th>';
    emotionsList.forEach(e => {
        headerRow += `<th>${capitalize(e)}</th>`;
    });
    headerRow += '</tr>';
    emotionsTable.innerHTML += headerRow;

    // Rows
    data.emotions_timeline.forEach(frame => {
        // Pour chaque frame, moyenne si plusieurs visages (ici tu peux adapter)
        let avgEmotions = {};
        emotionsList.forEach(e => avgEmotions[e] = 0);
        let facesCount = frame.detected_faces.length;

        frame.detected_faces.forEach(face => {
            emotionsList.forEach(e => {
                avgEmotions[e] += face.emotions[e] || 0;
            });
        });

        // Moyenne par émotion
        emotionsList.forEach(e => {
            avgEmotions[e] = facesCount > 0 ? avgEmotions[e] / facesCount : 0;
        });

        let row = `<tr><td>${frame.timestamp_seconds.toFixed(1)}</td>`;
        emotionsList.forEach(e => {
            row += `<td>${(avgEmotions[e] * 100).toFixed(1)}%</td>`;
        });
        row += '</tr>';
        emotionsTable.innerHTML += row;
    });

    // --- Construction du tableau sentiments ---
    const sentimentsTable = document.createElement('table');
    sentimentsTable.classList.add('result-table');

    // Header
    let sentimentHeader = '<tr><th>Temps (s)</th>';
    sentimentsList.forEach(s => {
        sentimentHeader += `<th>${capitalize(s)}</th>`;
    });
    sentimentHeader += '</tr>';
    sentimentsTable.innerHTML += sentimentHeader;

    // Rows
    data.emotions_timeline.forEach(frame => {
        let avgSentiments = {};
        sentimentsList.forEach(s => avgSentiments[s] = 0);
        let facesCount = frame.detected_faces.length;

        frame.detected_faces.forEach(face => {
            sentimentsList.forEach(s => {
                avgSentiments[s] += face.sentiments[s] || 0;
            });
        });

        sentimentsList.forEach(s => {
            avgSentiments[s] = facesCount > 0 ? avgSentiments[s] / facesCount : 0;
        });

        let row = `<tr><td>${frame.timestamp_seconds.toFixed(1)}</td>`;
        sentimentsList.forEach(s => {
            row += `<td>${(avgSentiments[s] * 100).toFixed(1)}%</td>`;
        });
        row += '</tr>';
        sentimentsTable.innerHTML += row;
    });

    // Ajout au container
    container.innerHTML = `<h4>Scores des émotions par frame</h4>`;
    container.appendChild(emotionsTable);
    container.innerHTML += `<h4>Scores des sentiments par frame</h4>`;
    container.appendChild(sentimentsTable);
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Global variables
let currentTab = 'transcribe';

// Tab switching functionality
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');

    currentTab = tabName;
}

// Helper function to get API base URL
function getApiUrl() {
    return document.getElementById('apiUrl').value.replace(/\/$/, '');
}

// Helper function to show loading
function showLoading(type) {
    document.getElementById(`${type}Loading`).classList.add('show');
}

// Helper function to hide loading
function hideLoading(type) {
    document.getElementById(`${type}Loading`).classList.remove('show');
}

// Helper function to display results
function displayResults(type, data) {
    const resultsDiv = document.getElementById(`${type}Results`);
    resultsDiv.innerHTML = `
        <div class="results">
            <h3>Results</h3>
            <pre>${JSON.stringify(data, null, 2)}</pre>
        </div>
    `;
}

// Helper function to show file info
function showFileInfo(file, type) {
    console.log(`${type} file selected:`, {
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: new Date(file.lastModified)
    });
}

// Helper function to display error
function displayError(type, message) {
    const resultsDiv = document.getElementById(`${type}Results`);
    resultsDiv.innerHTML = `
        <div class="error">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

// Audio transcription function
async function transcribeAudio() {
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an audio file first!');
        return;
    }

    showFileInfo(file, 'Audio');

    const formData = new FormData();
    formData.append('audio', file);

    showLoading('transcribe');

    try {
        const response = await fetch(`${getApiUrl()}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        displayResults('transcribe', result);
    } catch (error) {
        console.error('Transcription error:', error);
        displayError('transcribe', error.message);
    } finally {
        hideLoading('transcribe');
    }
}

// Face analysis function
async function analyzeFace() {
    const fileInput = document.getElementById('imageFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image file first!');
        return;
    }

    showFileInfo(file, 'Image');

    const formData = new FormData();
    formData.append('image', file);

    showLoading('face');

    try {
        const response = await fetch(`${getApiUrl()}/analyze-face`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        renderFaceImageTables(result);
    } catch (error) {
        console.error('Face analysis error:', error);
        displayError('face', error.message);
    } finally {
        hideLoading('face');
    }
}

// Video analysis function
async function analyzeVideo() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a video file first!');
        return;
    }

    showFileInfo(file, 'Video');

    const formData = new FormData();
    formData.append('video', file);

    showLoading('video');

    try {
        const response = await fetch(`${getApiUrl()}/analyze-video-emotions`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        displayVideoEmotionSentimentTables(result)
    } catch (error) {
        console.error('Video analysis error:', error);
        displayError('video', error.message);
    } finally {
        hideLoading('video');
    }
}

// Sentiment analysis function
async function analyzeSentiment() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();

    if (!text) {
        alert('Please enter some text first!');
        return;
    }

    const formData = new FormData();
    formData.append('text', text);

    showLoading('sentiment');

    try {
        const response = await fetch(`${getApiUrl()}/analyze-text-sentiment`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        displayResults('sentiment', result);
    } catch (error) {
        displayError('sentiment', error.message);
    } finally {
        hideLoading('sentiment');
    }
}

// Function to update upload area display
function updateUploadAreaDisplay(tabName, fileName) {
    const uploadArea = document.querySelector(`#${tabName} .upload-area`);
    const icons = {
        'transcribe': '🎵',
        'face': '😊', 
        'video': '🎬'
    };
    
    uploadArea.innerHTML = `
        <div class="upload-icon">${icons[tabName]}</div>
        <div class="upload-text">File selected: ${fileName}</div>
        <div class="upload-subtext">Click to change file or drag a new one</div>
    `;
}

// Function to reset upload area display
function resetUploadAreaDisplay(tabName) {
    const uploadArea = document.querySelector(`#${tabName} .upload-area`);
    const config = {
        'transcribe': {
            icon: '🎵',
            text: 'Drop your audio file here or click to browse',
            subtext: 'Supported: MP3, WAV, M4A, etc.'
        },
        'face': {
            icon: '😊',
            text: 'Drop your image here or click to browse',
            subtext: 'Supported: JPG, PNG, GIF, etc.'
        },
        'video': {
            icon: '🎬',
            text: 'Drop your video here or click to browse',
            subtext: 'Supported: MP4, MOV, AVI, etc. (Processing may take time)'
        }
    };
    
    const conf = config[tabName];
    uploadArea.innerHTML = `
        <div class="upload-icon">${conf.icon}</div>
        <div class="upload-text">${conf.text}</div>
        <div class="upload-subtext">${conf.subtext}</div>
    `;
}

// Drag and drop functionality
function setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        area.addEventListener('dragenter', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });

        area.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        area.addEventListener('dragleave', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
        });

        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                
                // Create a new FileList-like object
                const dt = new DataTransfer();
                dt.items.add(file);
                
                // Determine which input to use based on current tab and file type
                if (currentTab === 'transcribe' && file.type.startsWith('audio/')) {
                    document.getElementById('audioFile').files = dt.files;
                    updateUploadAreaDisplay('transcribe', file.name);
                } else if (currentTab === 'face' && file.type.startsWith('image/')) {
                    document.getElementById('imageFile').files = dt.files;
                    updateUploadAreaDisplay('face', file.name);
                } else if (currentTab === 'video' && file.type.startsWith('video/')) {
                    document.getElementById('videoFile').files = dt.files;
                    updateUploadAreaDisplay('video', file.name);
                } else {
                    alert(`Invalid file type for ${currentTab} analysis. Please select a valid ${currentTab === 'transcribe' ? 'audio' : currentTab === 'face' ? 'image' : 'video'} file.`);
                }
            }
        });
    });
}

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    setupDragAndDrop();

    // Add file change event listeners
    document.getElementById('audioFile').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            updateUploadAreaDisplay('transcribe', e.target.files[0].name);
        } else {
            resetUploadAreaDisplay('transcribe');
        }
    });

    document.getElementById('imageFile').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            updateUploadAreaDisplay('face', e.target.files[0].name);
        } else {
            resetUploadAreaDisplay('face');
        }
    });

    document.getElementById('videoFile').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            updateUploadAreaDisplay('video', e.target.files[0].name);
        } else {
            resetUploadAreaDisplay('video');
        }
    });

    // Add Enter key support for sentiment analysis
    document.getElementById('textInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeSentiment();
        }
    });
});
function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}