const API_BASE_URL = 'http://127.0.0.1:8000'; 
// For WebSocket, use 'ws://' for HTTP or 'wss://' for HTTPS
const WS_BASE_URL = 'ws://127.0.0.1:8000'; 
// pour la sauvegarde des données dans notre base:

let emotionTimeline = [];         // [{ timestamp, emotion, sentiment }]
let framesAnalyzed = 0;
let startTime = null;
let liveTextBuffer = "";
let lastTextEmotionResult = null;
let lastTextSentimentResult = null;



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
const sessionId = getSessionId();

// --- Elements for Webcam Analysis ---
const startWebcamButton = document.getElementById('startWebcamButton');
const stopWebcamButton = document.getElementById('stopWebcamButton');
const saveWebcamAnalysisButton = document.getElementById('saveWebcamAnalysisButton');
const saveWebcamAnalysisButtonUpload = document.getElementById('saveWebcamAnalysisButtonUpload')
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

const resultsSummaryDivLive = document.getElementById('resultsSummaryLive');
//const transcriptionTextElemLive = document.getElementById('transcriptionText'); // seulement si tu veux afficher la transcription quelque part
const facialEmotionsSummaryElemLive = document.getElementById('facialEmotionsSummaryLive');
const overallSentimentElemLive = document.getElementById('overallSentimentLive');
const rawSentimentScoresElemLive = document.getElementById('rawSentimentScoresLive');

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

// #########--- Uploaded Video Analysis Logic ---#########

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
    saveWebcamAnalysisButtonUpload.disabled = false;
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
        // Enregistrement automatique de currentAnalysisData dans un fichier JSON
        const analysisBlob = new Blob([JSON.stringify(currentAnalysisData, null, 2)], { type: 'application/json' });
        const analysisUrl = URL.createObjectURL(analysisBlob);

        const downloadLink = document.createElement('a');
        downloadLink.href = analysisUrl;
        downloadLink.download = 'analysis_result.json';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(analysisUrl); // Libère la mémoire
        const videoFileUrl = URL.createObjectURL(videoFile);
        setupUploadedVideoAndCanvas(videoFileUrl);

        //displayResultsSummary(currentAnalysisData);

    } catch (error) {
        console.error('Error during API call:', error);
        displayError(`Échec de l'analyse : ${error.message}`);
    } finally {
        loadingMessage.classList.add('hidden');
        analyzeButton.disabled = false;
    }
});

// ####### save reults from uploaded video #######

async function saveResults(currentAnalysisData, session_id) {
    try {
        // 📌 Sauvegarde de la transcription
        const transcriptionText = currentAnalysisData.transcription;
      if (transcriptionText) {
        await fetch('/api/save-transcription', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            session_id: session_id,
            transcription: transcriptionText
          })
        });
        console.log('✅ Transcription saved');
      }
    
        // 📌 Sauvegarde de l’analyse faciale
        const videoEmotions = currentAnalysisData.video_emotions;
        if (videoEmotions?.emotions_timeline) {
        const timeline = videoEmotions.emotions_timeline;

        const computeDominant = (timeline, key) => {
            const totals = {};
            timeline.forEach(frame => {
            frame.detected_faces.forEach(face => {
                const values = face[key];
                for (const label in values) {
                totals[label] = (totals[label] || 0) + values[label];
                }
            });
            });
            return Object.entries(totals).reduce((a, b) => a[1] > b[1] ? a : b, ['', 0])[0];
        };

        const dominantEmotion = computeDominant(timeline, 'emotions');
        const dominantSentiment = computeDominant(timeline, 'sentiments');

        // 🔧 Transformer le timeline au bon format
        const transformedTimeline = timeline.map(frame => ({
            timestamp: Math.round(frame.timestamp_seconds * 1000),
            detected_faces: frame.detected_faces.map(face => ({
            emotions: face.emotions,
            sentiments: face.sentiments
            }))
        }));

        // 🔄 Recréer l'objet video_emotions avec le nouveau timeline
        const transformedVideoEmotions = {
            ...videoEmotions,
            emotions_timeline: transformedTimeline
        };

        await fetch('/api/save-facial-emotions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
            session_id: session_id,
            dominant_emotion: dominantEmotion,
            dominant_sentiment: dominantSentiment,
            video_emotions: transformedVideoEmotions.emotions_timeline,
            duration_seconds: videoEmotions.video_duration_seconds,
            frames: videoEmotions.frames_analyzed
            })
        });

        console.log('✅ Facial analysis saved');
        }
        
    
        // 📌 Sauvegarde de l’analyse des sentiments du texte
        const textSentiment = currentAnalysisData.overall_text_sentiment;
        if (textSentiment) {
        await fetch('/api/save-text-sentiment', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json'
            },
            body: JSON.stringify({
            session_id: session_id,
            sentiment_label: textSentiment.overall_sentiment,
            confidence_score: textSentiment.confidence_score,
            raw_scores: textSentiment.raw_scores
            })
        });
    
        console.log('✅ Text sentiment saved');
        }
    
    } catch (error) {
        console.error('❌ Error saving results:', error);
    }
}
         
      
  

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

let audioRecorder;
let audioChunks = [];


async function startAudioTranscriptionRecorder(fullStream) {
    const audioTracks = fullStream.getAudioTracks();
    const audioStream = new MediaStream(audioTracks);

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "";

    if (!mimeType) {
        console.error("Aucun format audio compatible trouvé.");
        return;
    }

    const recorder = new MediaRecorder(audioStream, { mimeType });

    let chunks = [];

    recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
            console.log("🎤 Nouveau chunk audio reçu :", event.data);
            chunks.push(event.data);
        } else {
            console.warn("📭 Chunk vide ignoré");
        }
    };

    // Toutes les 5 secondes, on envoie le blob complet au serveur
    setInterval(async () => {
        if (chunks.length === 0) return;

        const completeBlob = new Blob(chunks, { type: mimeType });
        chunks = []; // reset les chunks

        const formData = new FormData();
        formData.append("file", completeBlob, "audio.webm");

        try {
            const res = await fetch("/transcribe_real_time", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errText = await res.text();
                console.error("❌ Erreur HTTP côté serveur :", errText);
                return;
            }

            const json = await res.json();
            //document.getElementById("liveTranscription").textContent += " " + json.text;
            displayLiveTranscription(json.text)
            liveTextBuffer += " " + json.text; 
        } catch (e) {
            console.error("❌ Transcription échouée :", e);
        }
    }, 5000);

    recorder.start();  // démarrage sans intervalle pour chunk
    console.log("🎙️ Enregistrement audio démarré, envoi toutes les 5 secondes");
}


async function analyzeLiveTextSummary() {
    if (!liveTextBuffer.trim()) {
        console.warn("⛔ Aucun texte à analyser");
        return;
    }

    const formData = new FormData();
    formData.append("text", liveTextBuffer);

    try {
        const [sentimentRes, emotionRes] = await Promise.all([
            fetch("/analyze-text-sentiment", {
                method: "POST",
                body: formData
            }),
            fetch("/analyze-text-emotion", {
                method: "POST",
                body: formData
            })
        ]);

        if (!sentimentRes.ok || !emotionRes.ok) {
            console.error("Erreur d’analyse de texte", await sentimentRes.text(), await emotionRes.text());
            return;
        }

        const sentimentJson = await sentimentRes.json();
        const emotionJson = await emotionRes.json();

        lastTextSentimentResult = sentimentJson;
        lastTextEmotionResult = emotionJson;



    } catch (err) {
        console.error("Erreur d’envoi vers les endpoints d’analyse :", err);
    }
}


/*async function startWebcamAnalysis() {
    console.log("🎬 Analyse en cours...");

    hideError();
    hideAllAnalysisDisplays(); 
    liveVideoContainer.classList.remove('hidden');
    liveEmotionFeedback.classList.remove('hidden');
    startWebcamButton.disabled = true;
    stopWebcamButton.disabled = false;
    isAnalyzingWebcam = true;
    startTime = Date.now();
    emotionTimeline = [];
    framesAnalyzed = 0;

    try {
        // Demande audio + vidéo
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });

        console.log("🎥 Webcam et 🎤 micro accessibles.");
        console.log("Pistes audio :", mediaStream.getAudioTracks());
        console.log("Pistes vidéo :", mediaStream.getVideoTracks());

        // ✅ Appel correct avec mediaStream
        startAudioTranscriptionRecorder(mediaStream);

        // Vidéo live
        webcamVideo.srcObject = mediaStream;
        webcamVideo.onloadedmetadata = () => {
            webcamAnalysisCanvas.width = webcamVideo.videoWidth;
            webcamAnalysisCanvas.height = webcamVideo.videoHeight;

            const aspectRatio = (webcamVideo.videoHeight / webcamVideo.videoWidth) * 100;
            document.querySelector('#liveVideoContainer .video-aspect-ratio').style.paddingBottom = `${aspectRatio}%`;

            webcamVideo.play();
            setupWebSocketAnalysis(); // Analyse temps réel via WebSocket
        };
    } catch (err) {
        console.error("❌ Erreur d'accès webcam/micro :", err);
        displayError("Impossible d'accéder à la webcam ou au micro. Vérifiez les autorisations.");
        stopWebcamAnalysis();
    }
}*/
let questions = [];
let currentQuestionIndex = 0;

// Références
const currentQuestionSpan = document.getElementById("currentQuestion");
const questionsDiv = document.getElementById("questions");
const nextQuestionButton = document.getElementById("nextQuestionButton"); // ➤ Tu dois ajouter ce bouton dans le HTML

async function startWebcamAnalysis() {
    console.log("🎬 Analyse en cours...");

    hideError();
    hideAllAnalysisDisplays();
    liveVideoContainer.classList.remove('hidden');
    liveEmotionFeedback.classList.remove('hidden');
    startWebcamButton.disabled = true;
    stopWebcamButton.disabled = false;
    isAnalyzingWebcam = true;
    startTime = Date.now();
    emotionTimeline = [];
    framesAnalyzed = 0;

    // 🎯 NOUVELLE PARTIE ➤ Récupérer les questions
    try {
        const res = await fetch('/questions/');
        if (!res.ok) throw new Error("Erreur de chargement des questions");
        questions = await res.json();
        currentQuestionIndex = 0;

        if (questions.length > 0) {
            questionsDiv.classList.remove('hidden');
            currentQuestionSpan.textContent = questions[0].question_text;
            nextQuestionButton.classList.remove('hidden');
            nextQuestionButton.disabled = false;
        } else {
            currentQuestionSpan.textContent = "Aucune question disponible.";
        }
    } catch (err) {
        console.error("❌ Erreur de récupération des questions :", err);
        displayError("Impossible de charger les questions.");
    }

    // 🎥 Vidéo/audio
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        webcamVideo.srcObject = mediaStream;
        webcamVideo.onloadedmetadata = () => {
            webcamAnalysisCanvas.width = webcamVideo.videoWidth;
            webcamAnalysisCanvas.height = webcamVideo.videoHeight;

            const aspectRatio = (webcamVideo.videoHeight / webcamVideo.videoWidth) * 100;
            document.querySelector('#liveVideoContainer .video-aspect-ratio').style.paddingBottom = `${aspectRatio}%`;

            webcamVideo.play();
            startAudioTranscriptionRecorder(mediaStream);
            setupWebSocketAnalysis();
        };
    } catch (err) {
        console.error("❌ Erreur d'accès webcam/micro :", err);
        displayError("Impossible d'accéder à la webcam ou au micro. Vérifiez les autorisations.");
        stopWebcamAnalysis();
    }
}

// ➤ Gérer clic sur “Suivant”
nextQuestionButton.addEventListener("click", () => {
    currentQuestionIndex++;
    if (currentQuestionIndex < questions.length) {
        currentQuestionSpan.textContent = questions[currentQuestionIndex].question_text;
    } else {
        currentQuestionSpan.textContent = "✅ Fin des questions.";
        nextQuestionButton.disabled = true;
    }
});



function displayLiveTranscription(text) {
    const transcriptionBox = document.getElementById("liveTranscription");
    if (transcriptionBox) {
        transcriptionBox.textContent += text + " ";
    }
}


/// helper function to get dominant values:

function getDominantValue(arr) {
    const counts = {};
    arr.forEach(v => { if (v) counts[v] = (counts[v]||0) + 1; });
    return Object.entries(counts).sort((a,b) => b[1] - a[1])[0]?.[0] || null;
}




let isStopping = false;

async function stopWebcamAnalysis() {
    
    console.log('stopWebcamAnalysis called');
    if (isStopping) {
        console.warn('⛔ Already stopping...');
        return;
    }
    isStopping = true;
    console.log('stopWebcamAnalysis called');
    console.log('✅ The analysis are stopping now !!!');

    stopWebcamButton.disabled = true;

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
    if (typeof window._stopAudioRecorder === 'function') {
        window._stopAudioRecorder();
    }    
    
    webcamCtx.clearRect(0, 0, webcamAnalysisCanvas.width, webcamAnalysisCanvas.height);
    liveVideoContainer.classList.add('hidden');
    liveEmotionFeedback.classList.add('hidden');
    startWebcamButton.disabled = false;
    console.log('✅ showing the button now now !!!');
    saveWebcamAnalysisButton.classList.remove('hidden');
    saveWebcamAnalysisButton.disabled = false;
    currentEmotionSpan.textContent = 'N/A';
    console.log('✅ dayen non ??!!!');

    setTimeout(() => {
        isStopping = false;
    }, 1000);

}



// Fonction qui sauvegarde les données d'analyse sur le serveur

/*async function saveWebcamAnalysis() {
    
    if (!sessionId) {
        console.error("❌ session_id non défini !");
        return;
    }

    if (emotionTimeline.length === 0) {
        console.error("❌ Pas de données à sauvegarder !");
        return;
    }
    showFinalEmotionAnalysisSummary()

    const durationSec = (Date.now() - startTime) / 1000;

    // Fonction de calcul de la dominante à partir des scores cumulés
    const computeDominant = (timeline, key) => {
        const totals = {};
        timeline.forEach(frame => {
            frame.detected_faces?.forEach(face => {
                const values = face[key];
                for (const label in values) {
                    totals[label] = (totals[label] || 0) + values[label];
                }
            });
        });
        return Object.entries(totals).reduce(
            (a, b) => a[1] > b[1] ? a : b,
            ['', 0]
        )[0];
    };

    // Calculer les dominantes à partir de tous les visages
    const dominantEmotion = computeDominant(emotionTimeline, 'emotions');
    const dominantSentiment = computeDominant(emotionTimeline, 'sentiments');

    try {
        await fetch('/api/save-webcam-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                dominant_emotion: dominantEmotion,
                dominant_sentiment: dominantSentiment,
                video_emotions: emotionTimeline, // toute la timeline dans un seul champ JSON
                duration_seconds: durationSec,
                frames_analyzed: framesAnalyzed
            })
        });
        console.log('✅ Analyse sauvegardée');
        saveWebcamAnalysisButton.disabled = true;
    } catch (err) {
        console.error('❌ Erreur sauvegarde :', err);
    }
    
}*/

async function saveWebcamAnalysis() {
    if (!sessionId) {
        console.error("❌ session_id non défini !");
        return;
    }

    if (emotionTimeline.length === 0) {
        console.error("❌ Pas de données à sauvegarder !");
        return;
    }

    showFinalEmotionAnalysisSummary();
    const token = localStorage.getItem('authToken');
    const durationSec = (Date.now() - startTime) / 1000;

    // Calcul des dominantes (émotions + sentiments)
    const computeDominant = (timeline, key) => {
        const totals = {};
        timeline.forEach(frame => {
            frame.detected_faces?.forEach(face => {
                const values = face[key];
                for (const label in values) {
                    totals[label] = (totals[label] || 0) + values[label];
                }
            });
        });
        return Object.entries(totals).reduce(
            (a, b) => a[1] > b[1] ? a : b,
            ['', 0]
        )[0];
    };

    const dominantEmotion = computeDominant(emotionTimeline, 'emotions');
    const dominantSentiment = computeDominant(emotionTimeline, 'sentiments');


    try {
        // Sauvegarde de l'analyse webcam habituelle
        await fetch('/api/save-webcam-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                dominant_emotion: dominantEmotion,
                dominant_sentiment: dominantSentiment,
                video_emotions: emotionTimeline,
                duration_seconds: durationSec,
                frames_analyzed: framesAnalyzed
            })
        });


    } catch (err) {
        console.error('❌ Erreur sauvegarde :', err);
    }
}



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
            if (analysisData.facial_emotions && Array.isArray(analysisData.facial_emotions)) {
                console.log(analysisData.facial_emotions)
                drawWebcamAnalysisOverlay(analysisData.facial_emotions);
                updateLiveEmotionFeedback(analysisData.facial_emotions);
                // Enregistrement des données
                const timestamp = Date.now();
                //const { topEmotion, topSentiment } = extractTopEmotionSentiment(analysisData.facial_emotions);
                //const emotions = analysisData.facial_emotions[0].emotions
                //const sentiments = analysisData.facial_emotions[0].sentiments
                const detectedFaces = analysisData.facial_emotions.map(face => ({
                    emotions: face.emotions,
                    sentiments: face.sentiments
                }));
                //emotionTimeline.push({ timestamp, emotions: emotions, sentiments: sentiments });
                emotionTimeline.push({ timestamp, detected_faces: detectedFaces });
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
        //stopWebcamAnalysis(); // Ensure all resources are cleaned up
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

async function showFinalEmotionAnalysisSummary() {
    const summaryContainer = document.getElementById("finalAnalysisSummary");
    await analyzeLiveTextSummary()
    summaryContainer.innerHTML = ""; // reset

    const title = document.createElement("h1");
    title.textContent = " ***** Résumé Final de l'Analyse *****";
    summaryContainer.appendChild(title);

    // 📘 Partie 1 : Analyse du texte
    const textSection = document.createElement("div");
    let textAnalysis = `<br><br><br><h2>***** Analyse du Texte *****</h2>`;

    if (lastTextSentimentResult) {
        const sentiment = lastTextSentimentResult.overall_sentiment;
        const confidence = (lastTextSentimentResult.confidence_score * 100).toFixed(1);
        textAnalysis += `<p>Le sentiment global exprimé dans le texte est <strong>${sentiment}</strong> (confiance : <strong>${confidence}%</strong>).</p>`;

        textAnalysis += "<p>Répartition des sentiments détectés :</p><ul>";
        for (const [label, score] of Object.entries(lastTextSentimentResult.raw_scores)) {
            textAnalysis += `<li>${label.charAt(0).toUpperCase() + label.slice(1)}: ${(score * 100).toFixed(1)}%</li>`;
        }
        textAnalysis += "</ul><br>";
    }

    if (lastTextEmotionResult) {
        const emotion = lastTextEmotionResult.overall_emotion;
        const confidence = (lastTextEmotionResult.confidence_score * 100).toFixed(1);
        textAnalysis += `<p>L'émotion principale détectée dans le discours est <strong>${emotion}</strong> (confiance : <strong>${confidence}%</strong>).</p>`;

        textAnalysis += "<p>Détail des émotions détectées :</p><ul>";
        for (const [label, score] of Object.entries(lastTextEmotionResult.raw_scores)) {
            textAnalysis += `<li>${label.charAt(0).toUpperCase() + label.slice(1)}: ${(score * 100).toFixed(1)}%</li>`;
        }
        textAnalysis += "</ul><br><br><br>";
    }

    textSection.innerHTML = textAnalysis;
    summaryContainer.appendChild(textSection);

    try {
        const token = localStorage.getItem("authToken");
        await fetch("/api/save-text-sentiment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: sessionId,  // ✅ Assure-toi que cette variable existe et contient l'ID de session
                sentiment_label: lastTextSentimentResult.overall_sentiment,
                confidence_score: lastTextSentimentResult.confidence_score,
                raw_scores: lastTextSentimentResult.raw_scores
            })
        });
    } catch (error) {
        console.error("❌ Erreur lors de l'enregistrement du sentiment :", error);
    }

    // 😶‍🌫️ Partie 2 : Analyse faciale synthétique
    const faceSection = document.createElement("div");
    faceSection.innerHTML = `<h2>***** Analyse des Émotions Faciales *****</h2>`;

    if (emotionTimeline.length === 0) {
        faceSection.innerHTML += `<p>Aucune donnée faciale détectée.</p><br>`;
    } else {
        const videoDurationSec = (emotionTimeline[emotionTimeline.length - 1].timestamp - emotionTimeline[0].timestamp) / 1000;
        let emotionCounts = {};
        let sentimentCounts = { Positive: 0, Neutral: 0, Negative: 0 };

        let altEmotions = {};
        let dominantEmotion = null;
        let maxEmotionCount = 0;

        emotionTimeline.forEach(entry => {
            entry.detected_faces.forEach(face => {
                // Emotion dominante
                const topEmotion = Object.entries(face.emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0];
                emotionCounts[topEmotion] = (emotionCounts[topEmotion] || 0) + 1;

                // Sentiment dominant
                const topSentiment = Object.entries(face.sentiments).reduce((a, b) => a[1] > b[1] ? a : b)[0];
                sentimentCounts[topSentiment] = (sentimentCounts[topSentiment] || 0) + 1;
            });
        });

        // Déterminer émotion dominante
        for (const [emo, count] of Object.entries(emotionCounts)) {
            if (count > maxEmotionCount) {
                maxEmotionCount = count;
                dominantEmotion = emo;
            }
        }

        //let faceSummary = `<p>Durant les ${Math.floor(videoDurationSec)} secondes analysées, plusieurs visages ont été détectés et évalués émotionnellement.</p>`;
        let faceSummary = `<p>Durant les ${videoDurationSec} secondes analysées, plusieurs visages ont été détectés et évalués émotionnellement.</p>`;
        if (dominantEmotion) {
            faceSummary += `<p>L’émotion faciale dominante observée était <strong>${dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1)}</strong>, détectée le plus fréquemment.</p>`;
        } else {
            faceSummary += `<p>Aucune émotion faciale dominante claire n'a émergé.</p>`;
        }

        if (Object.keys(emotionCounts).length > 1) {
            faceSummary += `<p>D’autres émotions ont aussi été relevées :<ul>`;
            for (const [emo, count] of Object.entries(emotionCounts)) {
                if (emo !== dominantEmotion) {
                    faceSummary += `<li>${emo.charAt(0).toUpperCase() + emo.slice(1)} : ${count} occurrences</li>`;
                }
            }
            faceSummary += `</ul></p><br>`;
        }

        // Analyse des sentiments faciaux
        const totalSentiments = sentimentCounts.Positive + sentimentCounts.Neutral + sentimentCounts.Negative;
        if (totalSentiments > 0) {
            faceSummary += `<p>Sentiment facial global : `;
            const dominantSentiment = Object.entries(sentimentCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0];
            faceSummary += `<strong>${dominantSentiment}</strong>.</p><br>`;
            faceSummary += `<p>Répartition approximative :<br>`;
            faceSummary += `&nbsp;&nbsp;• Positif : ${(sentimentCounts.Positive / totalSentiments * 100).toFixed(1)}%<br>`;
            faceSummary += `&nbsp;&nbsp;• Neutre : ${(sentimentCounts.Neutral / totalSentiments * 100).toFixed(1)}%<br>`;
            faceSummary += `&nbsp;&nbsp;• Négatif : ${(sentimentCounts.Negative / totalSentiments * 100).toFixed(1)}%</p>`;
        } else {
            faceSummary += `<p>Aucune donnée exploitable pour les sentiments visuels.</p>`;
        }

        faceSection.innerHTML += faceSummary;
    }

    summaryContainer.appendChild(faceSection);
        // 😓 Partie 3 : Analyse du Stress
    const stressSection = document.createElement("div");
    stressSection.innerHTML = `<h2>***** Analyse du Stress *****</h2>`;

    if (emotionTimeline.length === 0) {
        stressSection.innerHTML += `<p>Aucune donnée faciale disponible pour estimer le stress.</p>`;
    } else {
        const emotionSums = {
            angry: 0, disgust: 0, fear: 0, happy: 0, sad: 0, surprise: 0, neutral: 0
        };
        let totalFaces = 0;

        // Accumuler les émotions
        emotionTimeline.forEach(entry => {
            entry.detected_faces.forEach(face => {
                for (const [emo, score] of Object.entries(face.emotions)) {
                    if (emotionSums.hasOwnProperty(emo)) {
                        emotionSums[emo] += score;
                    }
                }
                totalFaces++;
            });
        });

        // Moyenne des émotions
        const emotionAverages = {};
        for (const emo in emotionSums) {
            emotionAverages[emo] = totalFaces > 0 ? emotionSums[emo] / totalFaces : 0;
        }

        // Pondération des émotions liées au stress
        const stressInfluence = {
            angry: 1.0,
            fear: 1.0,
            sad: 0.8,
            surprise: 0.6,
            disgust: 0.4,
            neutral: 0.0,
            happy: -0.5
        };

        let rawStress = 0;
        for (const emo in emotionAverages) {
            rawStress += emotionAverages[emo] * (stressInfluence[emo] || 0);
        }

        let stressPercent = Math.max(0, Math.min(100, (rawStress * 100))); // clamp entre 0 et 100
        stressPercent = stressPercent.toFixed(1);

        stressSection.innerHTML += `<p>Niveau estimé de stress facial : <strong>${stressPercent}%</strong>.</p>`;

        let stressFacial = parseFloat(stressPercent);
        let stressTextuel = 0;

        // Si analyse émotionnelle du texte disponible
        if (lastTextEmotionResult && lastTextEmotionResult.raw_scores) {
            const emotionScores = lastTextEmotionResult.raw_scores;
            
            const stressEmotionWeights = {
                anger: 1.0,
                fear: 1.0,
                sadness: 0.8,
                surprise: 0.6,
                disgust: 0.4,
                neutral: 0.0,
                joy: -0.5,
                happiness: -0.5
            };

            let weightedSum = 0;
            for (const [emo, score] of Object.entries(emotionScores)) {
                const weight = stressEmotionWeights[emo.toLowerCase()] || 0;
                weightedSum += score * weight;
            }

            stressTextuel = Math.max(0, Math.min(1, weightedSum)); // clamp 0-1
            stressTextuel = parseFloat((stressTextuel * 100).toFixed(1));
        }

        // Stress global combiné
        const stressGlobal = parseFloat((0.6 * stressFacial + 0.4 * stressTextuel).toFixed(1));
        // 👉 ENVOI AU SERVEUR
        try {
            const token = localStorage.getItem("authToken"); // ou autre méthode pour récupérer le token
            await fetch('/stress_analysis/', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    stress_facial: stressFacial,
                    stress_textuel: stressTextuel,
                    stress_global: stressGlobal
                })
            });
        } catch (error) {
            console.error("Erreur lors de l'envoi des données de stress :", error);
        }
        stressSection.innerHTML += `<p><strong>Stress facial estimé :</strong> ${stressFacial}%</p>`;
        stressSection.innerHTML += `<p><strong>Stress textuel estimé :</strong> ${stressTextuel}%</p>`;
        stressSection.innerHTML += `<p><strong>Niveau de stress global (fusion texte + visage) :</strong> <span style="font-size:1.2em;color:#c0392b">${stressGlobal}%</span></p>`;

        // Interprétation
        let interpretation = "";
        if (stressGlobal < 20) {
            interpretation = "Excellent calme général. Aucune tension notable détectée.";
        } else if (stressGlobal < 40) {
            interpretation = "Niveau de stress faible. Bonne stabilité émotionnelle.";
        } else if (stressGlobal < 60) {
            interpretation = "Stress modéré. À surveiller selon le contexte.";
        } else if (stressGlobal < 80) {
            interpretation = "Stress élevé. Des signes visibles de tension émotionnelle.";
        } else {
            interpretation = "Stress critique. Charge émotionnelle très importante détectée.";
        }

        stressSection.innerHTML += `<p>${interpretation}</p>`;

    }

    summaryContainer.appendChild(stressSection);
    // ✅ Envoi du contenu textuel accumulé (liveBuffer) à l’endpoint FastAPI
    try {
        const token = localStorage.getItem("authToken");
        await fetch("/api/save-transcription", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            },
            body: JSON.stringify({
                session_id: sessionId,
                transcription: liveTextBuffer
            })
        });
    } catch (error) {
        console.error("Erreur lors de l'envoi de la transcription :", error);
    }

    

    summaryContainer.classList.remove("hidden");
}


saveWebcamAnalysisButtonUpload.addEventListener('click', () => {
    saveResults(currentAnalysisData, sessionId);
});

// Initialize with Upload tab active
switchTab('uploadTab');

// --- Variables globales pour les éléments HTML ---
//let transcriptionTextElem, facialEmotionsSummaryElem, overallSentimentElem, rawSentimentScoresElem;
//let audioEmotionSummary, rawAudioEmotionScores;
//let resultsSummaryDiv;

/*document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const videoFile = document.getElementById('videoFile');
    const analyzeButton = document.getElementById('analyzeButton');
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    resultsSummaryDiv = document.getElementById('resultsSummary');
    transcriptionTextElem = document.getElementById('transcriptionText');
    facialEmotionsSummaryElem = document.getElementById('facialEmotionsSummary');
    overallSentimentElem = document.getElementById('overallSentiment');
    rawSentimentScoresElem = document.getElementById('rawSentimentScores');
    audioEmotionSummary = document.getElementById('audioEmotionSummary');
    rawAudioEmotionScores = document.getElementById('rawAudioEmotionScores');
    const uploadedVideoDisplayContainer = document.getElementById('uploadedVideoDisplayContainer');
    const interviewVideo = document.getElementById('interviewVideo');
    const analysisCanvas = document.getElementById('analysisCanvas');
    const saveWebcamAnalysisButtonUpload = document.getElementById('saveWebcamAnalysisButtonUpload');

    let currentAnalysisData = null;

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        const file = videoFile.files[0];
        if (!file) {
            alert('Veuillez sélectionner un fichier vidéo.');
            return;
        }

        loadingMessage.classList.remove('hidden');
        errorMessage.classList.add('hidden');
        resultsSummaryDiv.classList.add('hidden');
        uploadedVideoDisplayContainer.classList.add('hidden');
        saveWebcamAnalysisButtonUpload.disabled = true;

        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch('/analyze-interview-video', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Erreur lors de l\'analyse vidéo.');
            }

            const data = await response.json();
            currentAnalysisData = data;
            saveWebcamAnalysisButtonUpload.disabled = false;

            // --- Appel de la fonction d'affichage détaillé des résultats ---
            displayResultsSummary(data);

            // --- Affichage de la vidéo ---
            const videoURL = URL.createObjectURL(file);
            interviewVideo.src = videoURL;
            uploadedVideoDisplayContainer.classList.remove('hidden');

        } catch (error) {
            loadingMessage.classList.add('hidden');
            errorMessage.classList.remove('hidden');
            errorText.textContent = error.message;
        } finally {
            loadingMessage.classList.add('hidden');
        }
    });
});*/
