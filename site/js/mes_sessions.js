let emotionChartInstance = null;
let sentimentChartInstance = null;

async function fetchUserSessions(limit) {
    const token = localStorage.getItem('authToken');
    if (!token) {
        alert("Non connecté");
        window.location.href = "login.html";
        return;
    }

    try {
        const response = await fetch(`/api/user-sessions?limit=${limit}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error("Erreur de récupération des sessions.");
        }

        const data = await response.json();
        renderTable(data);
        renderEmotionChart(data);
        renderFacialSentimentChart(data);
    } catch (error) {
        alert(error.message);
    }
}

function renderTable(sessions) {
    const tbody = document.getElementById("sessionsTableBody");
    tbody.innerHTML = "";

    sessions.forEach(sess => {
        // Pour afficher plusieurs analyses, on va concaténer les valeurs avec des retours à la ligne
        const emotions = sess.facial_analyses.map(fa => fa.dominant_emotion || '-').join('\n');
        const sentimentsFacial = sess.facial_analyses.map(fa => fa.dominant_sentiment_facial || '-').join('\n');
        const sentimentsText = sess.text_sentiments.map(ts => ts.dominant_sentiment_text || '-').join('\n');
        const confidencesText = sess.text_sentiments
            .map(ts => ts.text_confidence !== null && ts.text_confidence !== undefined ? ts.text_confidence.toFixed(2) : '-')
            .join('\n');

        const tr = document.createElement("tr");

        const fields = [
            sess.session_id,
            new Date(sess.created_at).toLocaleString(),
            emotions || '-',
            sentimentsFacial || '-',
            sentimentsText || '-',
            confidencesText || '-'
        ];

        fields.forEach(val => {
            const td = document.createElement("td");
            // Utiliser <pre> pour garder les retours à la ligne visibles
            const pre = document.createElement("pre");
            pre.textContent = val;
            td.appendChild(pre);
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });
}


function renderEmotionChart(sessions) {
    if (!sessions.length) return;

    // Trier pour obtenir la dernière session
    const latestSession = sessions.reduce((a, b) =>
        new Date(a.created_at) > new Date(b.created_at) ? a : b
    );

    // Extraire toutes les émotions de tous les facial_analyses
    const allEmotions = {};
    let count = 0;

    latestSession.facial_analyses.forEach(fa => {
        //const videoEmotions = JSON.parse(fa.video_emotions || '[]');
        let videoEmotions;
        if (typeof fa.video_emotions === 'string') {
            try {
                videoEmotions = JSON.parse(fa.video_emotions);
            } catch (e) {
                console.error("Erreur de parsing JSON:", e);
                videoEmotions = [];
            }
        } else {
            videoEmotions = fa.video_emotions || [];
        }

        videoEmotions.forEach(frame => {
            frame.detected_faces.forEach(face => {
                const emotions = face.emotions;
                for (const [emotion, value] of Object.entries(emotions)) {
                    allEmotions[emotion] = (allEmotions[emotion] || 0) + value;
                }
                count++;
            });
        });
    });

    // Moyenne
    const labels = Object.keys(allEmotions);
    const data = labels.map(emotion => allEmotions[emotion] / count);

    const colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
        '#9966FF', '#FF9F40', '#C9CBCF', '#FF6666'
    ];

    if (emotionChartInstance) {
        emotionChartInstance.destroy();
    }

    const ctx = document.getElementById('emotionChart').getContext('2d');
    emotionChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, labels.length)
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right'
                },
                title: {
                    display: false
                }
            }
        }
    });
}

function renderFacialSentimentChart(sessions) {
    if (!sessions.length) return;

    const latestSession = sessions.reduce((a, b) =>
        new Date(a.created_at) > new Date(b.created_at) ? a : b
    );

    const allSentiments = {};
    let count = 0;

    latestSession.facial_analyses.forEach(fa => {
        //const videoEmotions = JSON.parse(fa.video_emotions || '[]');
        let videoEmotions;
        if (typeof fa.video_emotions === 'string') {
            try {
                videoEmotions = JSON.parse(fa.video_emotions);
            } catch (e) {
                console.error("Erreur de parsing JSON:", e);
                videoEmotions = [];
            }
        } else {
            videoEmotions = fa.video_emotions || [];
        }

        videoEmotions.forEach(frame => {
            frame.detected_faces.forEach(face => {
                const sentiments = face.sentiments;
                for (const [sentiment, value] of Object.entries(sentiments)) {
                    allSentiments[sentiment] = (allSentiments[sentiment] || 0) + value;
                }
                count++;
            });
        });
    });

    const labels = Object.keys(allSentiments);
    const data = labels.map(s => allSentiments[s] / count);

    const colors = [
        '#8E44AD', '#3498DB', '#2ECC71', '#F39C12',
        '#E74C3C', '#1ABC9C', '#34495E', '#95A5A6'
    ];

    if (sentimentChartInstance) {
        sentimentChartInstance.destroy();
    }

    const ctx = document.getElementById('sentimentChart').getContext('2d');
    sentimentChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, labels.length)
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                },
                title: {
                    display: false
                }
            }
        }
    });
}


document.getElementById("refreshBtn").addEventListener("click", () => {
    const limit = document.getElementById("sessionLimit").value;
    fetchUserSessions(limit);
});

document.getElementById("logoutBtn").addEventListener("click", () => {
    localStorage.removeItem("authToken");
    window.location.href = "login.html";
});

// Chargement initial
window.onload = () => {
    document.getElementById("refreshBtn").click();
};



