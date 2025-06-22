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
// Fonction qui arrête l'analyse ET la vidéo 
/*function stopWebcamAnalysis() {
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
    // Montrer le bouton "Enregistrer les résultats"
    console.log('✅ you must have the save botton now');
    saveWebcamAnalysisButton.classList.remove('hidden');
    saveWebcamAnalysisButton.disabled = false;
    
}*/

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