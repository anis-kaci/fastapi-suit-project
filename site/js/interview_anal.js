const uploadForm = document.getElementById('uploadForm');
const videoFileInput = document.getElementById('videoFile');
const analyzeButton = document.getElementById('analyzeButton');
const loadingMessage = document.getElementById('loadingMessage');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resultsDiv = document.getElementById('results');
const transcriptionTextElem = document.getElementById('transcriptionText');
const facialEmotionsSummaryElem = document.getElementById('facialEmotionsSummary');
const overallSentimentElem = document.getElementById('overallSentiment');
const rawSentimentScoresElem = document.getElementById('rawSentimentScores');

// Function to display error messages
function displayError(message) {
    errorMessage.classList.remove('hidden');
    errorText.textContent = message;
    resultsDiv.classList.add('hidden'); // Hide results if an error occurs
}

// Function to hide error messages
function hideError() {
    errorMessage.classList.add('hidden');
    errorText.textContent = '';
}

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent default form submission

    hideError(); // Clear any previous errors
    resultsDiv.classList.add('hidden'); // Hide previous results
    loadingMessage.classList.remove('hidden'); // Show loading spinner
    analyzeButton.disabled = true; // Disable button during analysis

    const videoFile = videoFileInput.files[0];

    if (!videoFile) {
        displayError('Veuillez sélectionner un fichier vidéo.');
        loadingMessage.classList.add('hidden');
        analyzeButton.disabled = false;
        return;
    }

    const formData = new FormData();
    formData.append('video', videoFile); // 'video' must match the FastAPI endpoint's parameter name

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze-interview-video', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Erreur du serveur: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error during API call:', error);
        displayError(`Échec de l'analyse : ${error.message}`);
    } finally {
        loadingMessage.classList.add('hidden'); // Hide loading spinner
        analyzeButton.disabled = false; // Re-enable button
    }
});

function displayResults(data) {
    resultsDiv.classList.remove('hidden');

    // 1. Transcription
    transcriptionTextElem.textContent = data.transcription.text;

    // 2. Facial Emotions & Sentiments
    const videoDuration = data.video_emotions.video_duration_seconds;
    const emotionsTimeline = data.video_emotions.emotions_timeline;

    // --- EMOTIONS ---
    let emotionTotals = {};
    let emotionDurations = {};
    const durationPerFrame = videoDuration / emotionsTimeline.length;

    // --- SENTIMENTS ---
    let sentimentTotals = {};
    let sentimentDurations = {};

    emotionsTimeline.forEach(frameData => {
        frameData.detected_faces.forEach(faceData => {
            const emotionProbs = faceData.emotions || {};
            const sentimentProbs = faceData.sentiments || {};

            // Emotions
            for (const emotion in emotionProbs) {
                emotionTotals[emotion] = (emotionTotals[emotion] || 0) + emotionProbs[emotion];
                if (emotionProbs[emotion] > 0.5) {
                    emotionDurations[emotion] = (emotionDurations[emotion] || 0) + durationPerFrame;
                }
            }

            // Sentiments
            for (const sentiment in sentimentProbs) {
                sentimentTotals[sentiment] = (sentimentTotals[sentiment] || 0) + sentimentProbs[sentiment];
                if (sentimentProbs[sentiment] > 0.5) {
                    sentimentDurations[sentiment] = (sentimentDurations[sentiment] || 0) + durationPerFrame;
                }
            }
        });
    });

    // Émotion dominante
    let dominantEmotion = getDominantLabel(emotionTotals);
    let dominantSentiment = getDominantLabel(sentimentTotals);

    let facialSummary = `Sur une durée de <strong>${videoDuration.toFixed(1)} secondes</strong>, en analysant ${emotionsTimeline.length} images clés (environ une par seconde) :<br>`;

    // Résumé émotions
    if (dominantEmotion) {
        facialSummary += `L'émotion faciale la plus fréquente était la <strong>${capitalize(dominantEmotion)}</strong>, détectée pendant environ <strong>${(emotionDurations[dominantEmotion] || 0).toFixed(1)} secondes</strong>.<br>`;
    } else {
        facialSummary += `Aucune émotion faciale dominante n'a pu être identifiée.<br>`;
    }

    // Résumé sentiments
    if (dominantSentiment) {
        facialSummary += `Le sentiment facial le plus prédominant était <strong>${capitalize(dominantSentiment)}</strong>, observé pendant environ <strong>${(sentimentDurations[dominantSentiment] || 0).toFixed(1)} secondes</strong>.<br>`;
    } else {
        facialSummary += `Aucun sentiment facial dominant n'a pu être identifié.<br>`;
    }

    // Détails émotions secondaires
    const threshold = 0.6;
    let keyEmotionTimestamps = {};
    let keySentimentTimestamps = {};

    emotionsTimeline.forEach(frameData => {
        const timestamp = frameData.timestamp_seconds;
        frameData.detected_faces.forEach(faceData => {
            const emotionProbs = faceData.emotions || {};
            const sentimentProbs = faceData.sentiments || {};

            for (const emo in emotionProbs) {
                if (emo !== dominantEmotion && emotionProbs[emo] >= threshold) {
                    keyEmotionTimestamps[emo] = keyEmotionTimestamps[emo] || new Set();
                    keyEmotionTimestamps[emo].add(timestamp);
                }
            }

            for (const sent in sentimentProbs) {
                if (sent !== dominantSentiment && sentimentProbs[sent] >= threshold) {
                    keySentimentTimestamps[sent] = keySentimentTimestamps[sent] || new Set();
                    keySentimentTimestamps[sent].add(timestamp);
                }
            }
        });
    });

    // Affichage des émotions secondaires
    if (Object.keys(keyEmotionTimestamps).length > 0) {
        facialSummary += `D'autres émotions notables ont été observées à certains moments clés :<br>`;
        for (const emo in keyEmotionTimestamps) {
            const timestamps = Array.from(keyEmotionTimestamps[emo]).sort((a, b) => a - b);
            const duration = (timestamps.length * durationPerFrame).toFixed(1);
            const formattedTimes = timestamps.map(t => t.toFixed(1)).join(', ');
            facialSummary += `- <strong>${capitalize(emo)}</strong> détectée pendant ~<strong>${duration} sec</strong> autour des secondes : <strong>${formattedTimes}</strong><br>`;
        }
    }

    // Affichage des sentiments secondaires
    if (Object.keys(keySentimentTimestamps).length > 0) {
        facialSummary += `<br>Des variations de sentiment facial ont également été détectées :<br>`;
        for (const sent in keySentimentTimestamps) {
            const timestamps = Array.from(keySentimentTimestamps[sent]).sort((a, b) => a - b);
            const duration = (timestamps.length * durationPerFrame).toFixed(1);
            const formattedTimes = timestamps.map(t => t.toFixed(1)).join(', ');
            facialSummary += `- <strong>${capitalize(sent)}</strong> observé pendant ~<strong>${duration} sec</strong> autour des secondes : <strong>${formattedTimes}</strong><br>`;
        }
    }

    facialEmotionsSummaryElem.innerHTML = facialSummary;

    // 3. Overall Sentiment from Transcript
    const overallSentiment = data.overall_text_sentiment.overall_sentiment;
    const confidenceScore = data.overall_text_sentiment.confidence_score;
    const rawScores = data.overall_text_sentiment.raw_scores;

    overallSentimentElem.innerHTML = `Sur la base de la transcription de votre réponse, le sentiment général détecté est <strong>${capitalize(overallSentiment)}</strong>, avec un score de confiance de <strong>${(confidenceScore * 100).toFixed(1)}%</strong>.`;

    rawSentimentScoresElem.innerHTML = '<p class="font-semibold mb-1">Détails des scores de sentiment :</p>';
    let scoresList = '<ul>';
    for (const label in rawScores) {
        scoresList += `<li>${capitalize(label)} : ${(rawScores[label] * 100).toFixed(1)}%</li>`;
    }
    scoresList += '</ul>';
    rawSentimentScoresElem.innerHTML += scoresList;
}

// Utilitaire : retourne l'étiquette avec le score le plus élevé
function getDominantLabel(totals) {
    let max = 0;
    let label = null;
    for (const key in totals) {
        if (totals[key] > max) {
            max = totals[key];
            label = key;
        }
    }
    return label;
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}