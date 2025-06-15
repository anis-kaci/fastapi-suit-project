const API_BASE_URL = 'http://127.0.0.1:8000'; 
// For WebSocket, use 'ws://' for HTTP or 'wss://' for HTTPS
const WS_BASE_URL = 'ws://127.0.0.1:8000'; 
// pour la sauvegarde des données dans notre base:

let emotionTimeline = [];         // [{ timestamp, emotion, sentiment }]
let framesAnalyzed = 0;
let startTime = null;



// --- Elements for Uploaded Video Analysis ---
const uploadTab = document.getElementById('uploadTab');
const webcamTab = document.getElementById('webcamTab');
const uploadSection = document.getElementById('uploadSection');
const webcamSection = document.getElementById('webcamSection');

const uploadForm = document.getElementById('uploadForm');
const videoFileInput = document.getElementById('videoFile');
const analyzeButton = document.getElementById('analyzeButton');
const uploadedVideoDisplayContainer = document.getElementById('uploadedVideoDisplayContainer');
const interviewVideo = document.getElementById('interviewVideo');
const analysisCanvas = document.getElementById('analysisCanvas');
const ctx = analysisCanvas.getContext('2d');
let currentAnalysisData = null; // Stores the full analysis data from the API
let animationFrameId = null; // For requestAnimationFrame on uploaded video


// --- Elements for Webcam Analysis ---
const startWebcamButton = document.getElementById('startWebcamButton');
const stopWebcamButton = document.getElementById('stopWebcamButton');
const saveWebcamAnalysisButton = document.getElementById('saveWebcamAnalysisButton');
const liveVideoContainer = document.getElementById('liveVideoContainer');
const webcamVideo = document.getElementById('webcamVideo');
const webcamAnalysisCanvas = document.getElementById('webcamAnalysisCanvas');
const webcamCtx = webcamAnalysisCanvas.getContext('2d');
const liveEmotionFeedback = document.getElementById('liveEmotionFeedback');
const currentEmotionSpan = document.getElementById('currentEmotion');
let mediaStream = null; // To store the webcam stream
let webcamAnalysisInterval = null; // For setInterval on webcam frames
let isAnalyzingWebcam = false;
let ws = null; // WebSocket object for real-time analysis


// --- Common Elements ---
const loadingMessage = document.getElementById('loadingMessage');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resultsSummaryDiv = document.getElementById('resultsSummary');
const transcriptionTextElem = document.getElementById('transcriptionText');
const facialEmotionsSummaryElem = document.getElementById('facialEmotionsSummary');
const overallSentimentElem = document.getElementById('overallSentiment');
const rawSentimentScoresElem = document.getElementById('rawSentimentScores');
// recuperer le session id

function getSessionId() {
    return localStorage.getItem('sessionId');
}



// --- Tab Switching Logic ---
function switchTab(activeTabId) {
    uploadTab.classList.remove('border-indigo-600', 'text-indigo-600');
    uploadTab.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700');
    webcamTab.classList.remove('border-indigo-600', 'text-indigo-600');
    webcamTab.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700');

    uploadSection.classList.add('hidden');
    webcamSection.classList.add('hidden');

    // Hide analysis results and video display when switching tabs
    hideAllAnalysisDisplays();

    if (activeTabId === 'uploadTab') {
        uploadTab.classList.add('border-indigo-600', 'text-indigo-600');
        uploadSection.classList.remove('hidden');
        // Ensure webcam is off if switching back
        stopWebcamAnalysis();
    } else if (activeTabId === 'webcamTab') {
        webcamTab.classList.add('border-indigo-600', 'text-indigo-600');
        webcamSection.classList.remove('hidden');
        // Prepare webcam section
        liveVideoContainer.classList.add('hidden');
        liveEmotionFeedback.classList.add('hidden');
        stopWebcamButton.disabled = true;
        startWebcamButton.disabled = false;
    }
}

uploadTab.addEventListener('click', () => switchTab('uploadTab'));
webcamTab.addEventListener('click', () => switchTab('webcamTab'));

function hideAllAnalysisDisplays() {
    loadingMessage.classList.add('hidden');
    errorMessage.classList.add('hidden');
    resultsSummaryDiv.classList.add('hidden');
    uploadedVideoDisplayContainer.classList.add('hidden');
    interviewVideo.src = '';
    ctx.clearRect(0, 0, analysisCanvas.width, analysisCanvas.height);
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    // For webcam specific displays
    liveVideoContainer.classList.add('hidden');
    liveEmotionFeedback.classList.add('hidden');
}

// --- Common Error/Loading/Display Functions ---
function displayError(message) {
    errorMessage.classList.remove('hidden');
    errorText.textContent = message;
    hideAllAnalysisDisplays(); // Hide everything else on error
}

function hideError() {
    errorMessage.classList.add('hidden');
    errorText.textContent = '';
}

// --- Uploaded Video Analysis Logic ---
function setupUploadedVideoAndCanvas(videoFileUrl) {
    uploadedVideoDisplayContainer.classList.remove('hidden');
    interviewVideo.src = videoFileUrl;

    interviewVideo.onloadedmetadata = () => {
        analysisCanvas.width = interviewVideo.videoWidth;
        analysisCanvas.height = interviewVideo.videoHeight;
        const aspectRatio = (interviewVideo.videoHeight / interviewVideo.videoWidth) * 100;
        document.querySelector('#uploadedVideoDisplayContainer .video-aspect-ratio').style.paddingBottom = `${aspectRatio}%`;
        interviewVideo.play();
    };

    interviewVideo.onplay = () => {
        drawUploadedVideoAnalysisLoop();
    };

    interviewVideo.onpause = () => {
        cancelAnimationFrame(animationFrameId);
    };
    interviewVideo.onended = () => {
        cancelAnimationFrame(animationFrameId);
        ctx.clearRect(0, 0, analysisCanvas.width, analysisCanvas.height);
    };
}

function drawUploadedVideoAnalysisLoop() {
    ctx.clearRect(0, 0, analysisCanvas.width, analysisCanvas.height);

    if (currentAnalysisData && interviewVideo.currentTime !== undefined) {
        const currentTime = Math.floor(interviewVideo.currentTime);
        const emotionsForCurrentSecond = currentAnalysisData.video_emotions.emotions_timeline.find(
            item => item.timestamp_seconds === currentTime
        );

        if (emotionsForCurrentSecond && emotionsForCurrentSecond.detected_faces.length > 0) {
            emotionsForCurrentSecond.detected_faces.forEach(face => {
                const bbox = face.bounding_box;
                const emotionProbs = face.emotions;
                const sentimentProbs = face.sentiments;

                // Emotion dominante
                let dominantEmotion = "";
                let confidence = 0;
                if (emotionProbs && Object.keys(emotionProbs).length > 0) {
                    [dominantEmotion, confidence] = Object.entries(emotionProbs).reduce(
                        (max, [emotion, prob]) => prob > max[1] ? [emotion, prob] : max,
                        ["", 0]
                    );
                }

                // Sentiment dominant
                let dominantSentiment = "";
                let sentimentConfidence = 0;
                if (sentimentProbs && Object.keys(sentimentProbs).length > 0) {
                    [dominantSentiment, sentimentConfidence] = Object.entries(sentimentProbs).reduce(
                        (max, [sentiment, prob]) => prob > max[1] ? [sentiment, prob] : max,
                        ["", 0]
                    );
                }

                const scaleX = analysisCanvas.width / interviewVideo.videoWidth;
                const scaleY = analysisCanvas.height / interviewVideo.videoHeight;

                const x = bbox.x * scaleX;
                const y = bbox.y * scaleY;
                const width = bbox.width * scaleX;
                const height = bbox.height * scaleY;

                ctx.strokeStyle = '#6366f1';
                ctx.lineWidth = 4;
                ctx.strokeRect(x, y, width, height);

                ctx.fillStyle = '#6366f1';
                const fontSize = Math.max(16, Math.floor(height / 8));
                ctx.font = `${fontSize}px Inter`;

                const label = `${dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1)} (${(confidence * 100).toFixed(1)}%) / ` +
                              `${dominantSentiment.charAt(0).toUpperCase() + dominantSentiment.slice(1)} (${(sentimentConfidence * 100).toFixed(1)}%)`;

                const textWidth = ctx.measureText(label).width;
                const labelX = x + (width - textWidth) / 2;

                ctx.fillText(label, labelX, y - 10);
            });
        }
    }

    animationFrameId = requestAnimationFrame(drawUploadedVideoAnalysisLoop);
}



function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}


function drawUploadedImageAnalysis(data) {
    ctx.clearRect(0, 0, analysisCanvas.width, analysisCanvas.height);

    const faces = data.facial_emotions;
    faces.forEach(face => {
        const bbox = face.bounding_box;
        const emotions = face.emotions;
        const sentiments = face.sentiments;

        // 1. Trouver l’émotion dominante
        let dominantEmotion = Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b);
        let emotionLabel = dominantEmotion[0];
        let emotionScore = dominantEmotion[1];

        // 2. Trouver le sentiment dominant
        let dominantSentiment = Object.entries(sentiments).reduce((a, b) => a[1] > b[1] ? a : b);
        let sentimentLabel = dominantSentiment[0];
        let sentimentScore = dominantSentiment[1];

        // Adapter la taille si nécessaire
        const scaleX = analysisCanvas.width / interviewVideo.videoWidth;
        const scaleY = analysisCanvas.height / interviewVideo.videoHeight;

        const x = bbox.x * scaleX;
        const y = bbox.y * scaleY;
        const width = bbox.width * scaleX;
        const height = bbox.height * scaleY;

        // Dessiner le rectangle
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        // Texte affiché
        const fontSize = Math.max(16, Math.floor(height / 8));
        ctx.font = `${fontSize}px Inter`;
        ctx.fillStyle = '#6366f1';

        const emotionText = `${capitalize(emotionLabel)} (${(emotionScore * 100).toFixed(1)}%)`;
        const sentimentText = `Sentiment: ${capitalize(sentimentLabel)} (${(sentimentScore * 100).toFixed(1)}%)`;

        ctx.fillText(emotionText, x + 5, y - 15);
        ctx.fillText(sentimentText, x + 5, y - 15 - fontSize);
    });
}


uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    hideError();
    hideAllAnalysisDisplays(); // Ensure all displays are hidden
    loadingMessage.classList.remove('hidden');
    analyzeButton.disabled = true;

    const videoFile = videoFileInput.files[0];

    if (!videoFile) {
        displayError('Veuillez sélectionner un fichier vidéo.');
        loadingMessage.classList.add('hidden');
        analyzeButton.disabled = false;
        return;
    }

    const formData = new FormData();
    formData.append('video', videoFile);

    try {
        const response = await fetch(`${API_BASE_URL}/analyze-interview-video`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Erreur du serveur: ${response.status} ${response.statusText}`);
        }

        currentAnalysisData = await response.json(); // Store the full analysis data
        
        const videoFileUrl = URL.createObjectURL(videoFile);
        setupUploadedVideoAndCanvas(videoFileUrl);

        displayResultsSummary(currentAnalysisData);

    } catch (error) {
        console.error('Error during API call:', error);
        displayError(`Échec de l'analyse : ${error.message}`);
    } finally {
        loadingMessage.classList.add('hidden');
        analyzeButton.disabled = false;
    }
});

// helper function to save some data

function extractTopEmotionSentiment(faces) {
    let topEmotion = '', emoConf = 0;
    let topSentiment = '', sentConf = 0;

    faces.forEach(face => {
        if (face.emotions) {
            const [e, c] = Object.entries(face.emotions).reduce((m, cur) => cur[1] > m[1] ? cur : m, ["","0"]);
            if (c > emoConf) { emoConf = c; topEmotion = e; }
        }
        if (face.sentiments) {
            const [s, c] = Object.entries(face.sentiments).reduce((m, cur) => cur[1] > m[1] ? cur : m, ["","0"]);
            if (c > sentConf) { sentConf = c; topSentiment = s; }
        }
    });

    return { topEmotion, topSentiment };
}



// --- Webcam Analysis Logic (UPDATED FOR WEBSOCKET) ---
startWebcamButton.addEventListener('click', startWebcamAnalysis);
stopWebcamButton.addEventListener('click', stopWebcamAnalysis);
saveWebcamAnalysisButton.addEventListener('click', saveWebcamAnalysis);

async function startWebcamAnalysis() {
    console.log('analyse en cours hihihi ...');

    hideError();
    hideAllAnalysisDisplays(); 
    liveVideoContainer.classList.remove('hidden');
    liveEmotionFeedback.classList.remove('hidden');
    startWebcamButton.disabled = true;
    stopWebcamButton.disabled = false;
    isAnalyzingWebcam = true; // Set flag for ongoing analysis
    startTime = Date.now();
    emotionTimeline = [];
    framesAnalyzed = 0;

    try {
        // Request both video and audio streams
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true }); 
        webcamVideo.srcObject = mediaStream;
        webcamVideo.onloadedmetadata = () => {
            webcamAnalysisCanvas.width = webcamVideo.videoWidth;
            webcamAnalysisCanvas.height = webcamVideo.videoHeight;
            const aspectRatio = (webcamVideo.videoHeight / webcamVideo.videoWidth) * 100;
            document.querySelector('#liveVideoContainer .video-aspect-ratio').style.paddingBottom = `${aspectRatio}%`;
            webcamVideo.play();
            setupWebSocketAnalysis(); // Initiate WebSocket connection
        };
    } catch (err) {
        console.error("Erreur d'accès à la webcam : ", err);
        displayError("Impossible d'accéder à la webcam. Assurez-vous d'avoir autorisé l'accès et qu'aucune autre application ne l'utilise.");
        stopWebcamAnalysis(); // Reset buttons and state
    }
}



/// helper function to get dominant values:

function getDominantValue(arr) {
    const counts = {};
    arr.forEach(v => { if (v) counts[v] = (counts[v]||0) + 1; });
    return Object.entries(counts).sort((a,b) => b[1] - a[1])[0]?.[0] || null;
}


/*async function stopWebcamAnalysis() {
    console.log('✅ the analysis are stopping now !!!');
    const sessionId = getSessionId();
    if (!sessionId) {
        console.error("❌ session_id non défini !");
        return;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
    if (webcamAnalysisInterval) {
        clearInterval(webcamAnalysisInterval);
        webcamAnalysisInterval = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(); // Close WebSocket connection
    }
    webcamCtx.clearRect(0, 0, webcamAnalysisCanvas.width, webcamAnalysisCanvas.height); // Clear canvas
    liveVideoContainer.classList.add('hidden');
    liveEmotionFeedback.classList.add('hidden');
    startWebcamButton.disabled = false;
    stopWebcamButton.disabled = true;
    isAnalyzingWebcam = false;
    currentEmotionSpan.textContent = 'N/A';

    const durationSec = (Date.now() - startTime) / 1000;
    const dominantEmotion = getDominantValue(emotionTimeline.map(e => e.emotion));
    const dominantSentiment = getDominantValue(emotionTimeline.map(e => e.sentiment));

    try {
        await fetch('/api/save-webcam-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId, 
                dominant_emotion: dominantEmotion,
                emotion_timeline: emotionTimeline,
                dominant_sentiment: dominantSentiment,
                sentiment_timeline: emotionTimeline.map(e => ({ timestamp: e.timestamp, sentiment: e.sentiment })),
                duration_seconds: durationSec,
                frames: framesAnalyzed
            })
        });
        console.log('✅ Analyse sauvegardée');
    } catch (err) {
        console.error('❌ Erreur sauvegarde :', err);
    }
}*/
// Fonction qui arrête l'analyse ET la vidéo (mais sans sauvegarder)
function stopWebcamAnalysis() {
    console.log('✅ The analysis are stopping now !!!');
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
    if (webcamAnalysisInterval) {
        clearInterval(webcamAnalysisInterval);
        webcamAnalysisInterval = null;
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
    webcamCtx.clearRect(0, 0, webcamAnalysisCanvas.width, webcamAnalysisCanvas.height);
    liveVideoContainer.classList.add('hidden');
    liveEmotionFeedback.classList.add('hidden');
    startWebcamButton.disabled = false;
    stopWebcamButton.disabled = true;
    saveWebcamAnalysisButton.disabled = false;  // Activer bouton sauvegarde
    isAnalyzingWebcam = false;
    currentEmotionSpan.textContent = 'N/A';
}

// Fonction qui sauvegarde les données d'analyse sur le serveur
async function saveWebcamAnalysis() {
    const sessionId = getSessionId();
    if (!sessionId) {
        console.error("❌ session_id non défini !");
        return;
    }

    if (emotionTimeline.length === 0) {
        console.error("❌ Pas de données à sauvegarder !");
        return;
    }

    const durationSec = (Date.now() - startTime) / 1000;
    const dominantEmotion = getDominantValue(emotionTimeline.map(e => e.emotion));
    const dominantSentiment = getDominantValue(emotionTimeline.map(e => e.sentiment));

    try {
        await fetch('/api/save-webcam-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                dominant_emotion: dominantEmotion,
                emotion_timeline: emotionTimeline,
                dominant_sentiment: dominantSentiment,
                sentiment_timeline: emotionTimeline.map(e => ({ timestamp: e.timestamp, sentiment: e.sentiment })),
                duration_seconds: durationSec,
                frames: framesAnalyzed
            })
        });
        console.log('✅ Analyse sauvegardée');
        saveWebcamAnalysisButton.disabled = true; // désactiver bouton sauvegarde après réussite
    } catch (err) {
        console.error('❌ Erreur sauvegarde :', err);
    }
}


/*async function stopWebcamAnalysis() {
    const endTime = Date.now();
    const durationSeconds = (endTime - startTime) / 1000;

    const dominantEmotion = getDominantValue(emotionTimeline.map(e => e.emotion));
    const dominantSentiment = getDominantValue(emotionTimeline.map(e => e.sentiment));

    const sentimentTimeline = emotionTimeline.map(e => ({
        timestamp: e.timestamp,
        sentiment: e.sentiment
    }));

    const payload = {
        session_id: currentSessionId,  // À remplacer par ta vraie variable de session
        dominant_emotion: dominantEmotion,
        emotion_timeline: emotionTimeline,
        dominant_sentiment: dominantSentiment,
        sentiment_timeline: sentimentTimeline,
        duration: durationSeconds,
        frames: framesAnalyzed
    };

    try {
        const response = await fetch("/api/save-webcam-analysis", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!response.ok) throw new Error(result.detail);
        console.log("✅ Analyse sauvegardée :", result.message);
    } catch (error) {
        console.error("❌ Erreur lors de la sauvegarde :", error.message);
    }
}*/

function setupWebSocketAnalysis() {
    // Establish WebSocket connection
    ws = new WebSocket(`${WS_BASE_URL}/ws/analyze-webcam`); // This endpoint needs to be implemented in your FastAPI backend

    ws.onopen = () => {
        console.log("WebSocket connected for real-time analysis.");
        startSendingFramesOverWebSocket(); // Start sending frames once connected
    };

    ws.onmessage = (event) => {
        // Parse the JSON data sent from the backend
        try {
            const analysisData = JSON.parse(event.data);
            console.log("Received real-time analysis:", analysisData); // For debugging: CHECK THIS OUTPUT!

           // if (analysisData.facial_emotions && Array.isArray(analysisData.facial_emotions.detected_faces)) {
           //     drawWebcamAnalysisOverlay(analysisData.facial_emotions.detected_faces);
           //     updateLiveEmotionFeedback(analysisData.facial_emotions.detected_faces);
           //} else if (analysisData.detail) { // Handle potential error messages from backend
           //     console.warn("Backend sent an error/info message:", analysisData.detail);
                // Optionally display a temporary message to the user
            //}
            if (analysisData.facial_emotions && Array.isArray(analysisData.facial_emotions)) {
                drawWebcamAnalysisOverlay(analysisData.facial_emotions);
                updateLiveEmotionFeedback(analysisData.facial_emotions);
                // Enregistrement des données
                const timestamp = Date.now();
                const { topEmotion, topSentiment } = extractTopEmotionSentiment(analysisData.facial_emotions);
                emotionTimeline.push({ timestamp, emotion: topEmotion, sentiment: topSentiment });
                framesAnalyzed++;
            } else if (analysisData.detail) { // Handle potential error messages from backend
                console.warn("Backend sent an error/info message:", analysisData.detail);
                // Optionally display a temporary message to the user
            }
            
            
        } catch (e) {
            console.error("Failed to parse WebSocket message:", e, event.data);
        }
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        displayError("Erreur de connexion WebSocket. L'analyse en temps réel est interrompue.");
        stopWebcamAnalysis(); // Stop analysis on error
    };

    ws.onclose = (event) => {
        console.log("WebSocket closed:", event);
        // Only show error if it wasn't intentionally stopped by the user
        if (isAnalyzingWebcam) { 
            displayError("Connexion à l'analyse en temps réel perdue.");
        }
        stopWebcamAnalysis(); // Ensure all resources are cleaned up
    };
}

function startSendingFramesOverWebSocket() {
    const frameSendIntervalMs = 200; // Send frames every 200ms (5 frames/sec) for visual analysis

    // Clear previous interval if any
    if (webcamAnalysisInterval) {
        clearInterval(webcamAnalysisInterval);
    }

    webcamAnalysisInterval = setInterval(() => {
        // Ensure webcam is active and WebSocket is open
        if (!isAnalyzingWebcam || webcamVideo.paused || webcamVideo.ended || ws.readyState !== WebSocket.OPEN) {
            return;
        }

        // Make sure video dimensions are available before drawing
        if (webcamVideo.videoWidth === 0 || webcamVideo.videoHeight === 0) {
            console.warn("Webcam video dimensions not available yet. Skipping frame.");
            return;
        }

        // Create a temporary canvas to draw the current webcam frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = webcamVideo.videoWidth;
        tempCanvas.height = webcamVideo.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw the video frame onto the temporary canvas
        tempCtx.drawImage(webcamVideo, 0, 0, tempCanvas.width, tempCanvas.height);

        // Convert the canvas content to a JPEG Blob for sending
        tempCanvas.toBlob((blob) => {
            if (blob) {
                ws.send(blob); // Send the image blob over WebSocket
            }
        }, 'image/jpeg', 0.8); // Specify image format (JPEG) and quality

        // TODO: For audio analysis, you would also need to capture audio chunks
        // from mediaStream and send them over the same (or a separate) WebSocket.
        // This typically involves using MediaRecorder API to record small audio segments.

    }, frameSendIntervalMs);
}

function drawWebcamAnalysisOverlay(detectedFaces) {
    webcamCtx.clearRect(0, 0, webcamAnalysisCanvas.width, webcamAnalysisCanvas.height);

    // Dessiner la vidéo sans miroir
    webcamCtx.drawImage(webcamVideo, 0, 0, webcamAnalysisCanvas.width, webcamAnalysisCanvas.height);

    detectedFaces.forEach(face => {
        const bbox = face.bounding_box;

        // Trouver l'émotion dominante
        let dominantEmotion = '';
        let emotionConfidence = 0;
        if (face.emotions && Object.keys(face.emotions).length > 0) {
            [dominantEmotion, emotionConfidence] = Object.entries(face.emotions).reduce(
                (max, [emotion, prob]) => prob > max[1] ? [emotion, prob] : max,
                ["", 0]
            );
        }

        // Trouver le sentiment dominant
        let dominantSentiment = '';
        let sentimentConfidence = 0;
        if (face.sentiments && Object.keys(face.sentiments).length > 0) {
            [dominantSentiment, sentimentConfidence] = Object.entries(face.sentiments).reduce(
                (max, [sentiment, prob]) => prob > max[1] ? [sentiment, prob] : max,
                ["", 0]
            );
        }

        const x = bbox.x;
        const y = bbox.y;
        const width = bbox.width;
        const height = bbox.height;

        // Rectangle bleu pour le visage
        webcamCtx.strokeStyle = '#3b82f6';
        webcamCtx.lineWidth = 4;
        webcamCtx.strokeRect(x, y, width, height);

        // Texte émotion en bleu
        webcamCtx.fillStyle = '#3b82f6';
        //webcamCtx.font = `${Math.max(18, Math.floor(height / 6))}px Inter`;
        webcamCtx.font = `${Math.max(12, Math.floor(height / 8))}px Inter`;
        webcamCtx.fillText(
            `${capitalize(dominantEmotion)} (${(emotionConfidence * 100).toFixed(1)}%)`, 
            x + 5, y - 35
        );

        // Texte sentiment en vert (un peu plus bas)
        if (dominantSentiment) {
            webcamCtx.fillStyle = '#22c55e'; // vert
            webcamCtx.fillText(
                `${capitalize(dominantSentiment)} (${(sentimentConfidence * 100).toFixed(1)}%)`, 
                x + 5, y - 5
            );
        }
    });
}


function updateLiveEmotionFeedback(detectedFaces) {
    if (detectedFaces.length > 0) {
        let topEmotion = '';
        let topEmotionConfidence = 0;
        let topSentiment = '';
        let topSentimentConfidence = 0;

        detectedFaces.forEach(face => {
            if (face.emotions) {
                const [dominantEmotion, confidence] = Object.entries(face.emotions).reduce(
                    (max, [emotion, prob]) => prob > max[1] ? [emotion, prob] : max,
                    ["", 0]
                );

                if (confidence > topEmotionConfidence) {
                    topEmotionConfidence = confidence;
                    topEmotion = dominantEmotion;
                }
            }
            if (face.sentiments) {
                const [dominantSentiment, confidence] = Object.entries(face.sentiments).reduce(
                    (max, [sentiment, prob]) => prob > max[1] ? [sentiment, prob] : max,
                    ["", 0]
                );

                if (confidence > topSentimentConfidence) {
                    topSentimentConfidence = confidence;
                    topSentiment = dominantSentiment;
                }
            }
        });

        currentEmotionSpan.textContent = 
            `Emotion: ${capitalize(topEmotion)} (${(topEmotionConfidence * 100).toFixed(1)}%) — ` +
            `Sentiment: ${capitalize(topSentiment)} (${(topSentimentConfidence * 100).toFixed(1)}%)`;
    } else {
        currentEmotionSpan.textContent = 'Aucun visage détecté';
    }
}





// --- Display Text Summary Results for Uploaded Videos ---
function displayResultsSummary(data) {
    resultsSummaryDiv.classList.remove('hidden');

    // 1. Transcription
    transcriptionTextElem.textContent = data.transcription.text;

    // 2. Facial Emotions Summary (for uploaded video)
    const videoDuration = data.video_emotions.video_duration_seconds;
    const emotionsTimeline = data.video_emotions.emotions_timeline;

    let facialSummary = `Sur une durée de <strong>${videoDuration} secondes</strong>, en analysant ${emotionsTimeline.length} images clés (une par seconde) : `;

    let emotionCounts = {};
    let dominantEmotion = null;
    let maxCount = 0;

    // Pour détecter la dominante et la fréquence des émotions
    emotionsTimeline.forEach(frameData => {
        frameData.detected_faces.forEach(faceData => {
            const emotionDistribution = faceData.emotions;
            // Trouver l’émotion avec le score le plus élevé
            const dominant = Object.entries(emotionDistribution).reduce((a, b) => a[1] > b[1] ? a : b)[0];
            emotionCounts[dominant] = (emotionCounts[dominant] || 0) + 1;
        });
    });

    // Déterminer l'émotion dominante globale
    for (const emotion in emotionCounts) {
        if (emotionCounts[emotion] > maxCount) {
            maxCount = emotionCounts[emotion];
            dominantEmotion = emotion;
        }
    }

    if (dominantEmotion) {
        facialSummary += `L'émotion faciale la plus fréquemment détectée était la <strong>${dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1)}</strong>.<br>`;
    } else {
        facialSummary += `Aucun visage significatif n'a été détecté ou aucune émotion dominante claire.<br>`;
    }

    // Détection d'autres émotions secondaires
    let keyEmotionTimestamps = {};
    emotionsTimeline.forEach(frameData => {
        frameData.detected_faces.forEach(faceData => {
            const emotionDistribution = faceData.emotions;
            const dominant = Object.entries(emotionDistribution).reduce((a, b) => a[1] > b[1] ? a : b)[0];

            if (dominantEmotion && dominant !== dominantEmotion) {
                if (!keyEmotionTimestamps[dominant]) {
                    keyEmotionTimestamps[dominant] = new Set();
                }
                keyEmotionTimestamps[dominant].add(frameData.timestamp_seconds);
            }
        });
    });

    if (Object.keys(keyEmotionTimestamps).length > 0) {
        facialSummary += "Cependant, il est important de noter des moments où d'autres émotions ont été observées :<br>";
        for (const emo in keyEmotionTimestamps) {
            const sortedTimestamps = Array.from(keyEmotionTimestamps[emo]).sort((a, b) => a - b).join(', ');
            facialSummary += `&nbsp;&nbsp;• Des expressions de <strong>${emo.charAt(0).toUpperCase() + emo.slice(1)}</strong> ont été détectées autour des secondes : <strong>${sortedTimestamps}</strong>.<br>`;
        }
        facialSummary += "<br>Ces variations émotionnelles sont intéressantes à considérer en regard du discours.<br>";
    } else if (dominantEmotion) {
        facialSummary += `<br>L'émotion <strong>${dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1)}</strong> a été prédominante tout au long des moments analysés.<br>`;
    }
    facialEmotionsSummaryElem.innerHTML = facialSummary;
    // 2b. Analyse des sentiments faciaux (visuels)
    let visualSentimentCounts = { Positive: 0, Neutral: 0, Negative: 0 };

    emotionsTimeline.forEach(frameData => {
        frameData.detected_faces.forEach(faceData => {
            const sentiments = faceData.sentiments;
            // Identifier le sentiment dominant pour ce visage
            const dominantSentiment = Object.entries(sentiments).reduce((a, b) => a[1] > b[1] ? a : b)[0];
            visualSentimentCounts[dominantSentiment] = (visualSentimentCounts[dominantSentiment] || 0) + 1;
        });
    });

    let totalSentimentDetections = visualSentimentCounts.Positive + visualSentimentCounts.Neutral + visualSentimentCounts.Negative;

    //let visualSentimentSummary = "<br><br><u>Analyse des sentiments visuels (à partir des visages détectés) :</u><br>";
    let visualSentimentSummary = "<br><br><u>Analyse des sentiments visuels:</u><br>";

    if (totalSentimentDetections > 0) {
        const mostFrequentSentiment = Object.entries(visualSentimentCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0];
        visualSentimentSummary += `Le sentiment visuel dominant était <strong>${mostFrequentSentiment}</strong>.<br>`;
        visualSentimentSummary += `Répartition approximative :<br>`;
        visualSentimentSummary += `&nbsp;&nbsp;• Positif : ${((visualSentimentCounts.Positive / totalSentimentDetections) * 100).toFixed(1)}%<br>`;
        visualSentimentSummary += `&nbsp;&nbsp;• Neutre : ${((visualSentimentCounts.Neutral / totalSentimentDetections) * 100).toFixed(1)}%<br>`;
        visualSentimentSummary += `&nbsp;&nbsp;• Négatif : ${((visualSentimentCounts.Negative / totalSentimentDetections) * 100).toFixed(1)}%<br>`;
    } else {
        visualSentimentSummary += "Aucune donnée exploitable pour les sentiments visuels.";
    }

    facialEmotionsSummaryElem.innerHTML += visualSentimentSummary;

    

    // 3. Overall Sentiment
    const overallSentiment = data.overall_text_sentiment.overall_sentiment;
    const confidenceScore = data.overall_text_sentiment.confidence_score;
    const rawScores = data.overall_text_sentiment.raw_scores;

    overallSentimentElem.innerHTML = `Sur la base de la transcription de votre réponse, le sentiment général détecté est <strong>${overallSentiment.charAt(0).toUpperCase() + overallSentiment.slice(1)}</strong>, avec un score de confiance de <strong>${(confidenceScore * 100).toFixed(1)}%</strong>.`;

    rawSentimentScoresElem.innerHTML = '<p class="font-semibold mb-1">Détails des scores de sentiment :</p>';
    let scoresList = '<ul>';
    for (const label in rawScores) {
        scoresList += `<li>${label.charAt(0).toUpperCase() + label.slice(1)}: ${(rawScores[label] * 100).toFixed(1)}%</li>`;
    }
    scoresList += '</ul>';
    rawSentimentScoresElem.innerHTML += scoresList;
}

// Initialize with Upload tab active
switchTab('uploadTab');