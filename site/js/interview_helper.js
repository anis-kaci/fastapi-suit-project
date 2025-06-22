/*function switchTab(tabId) {
    // Remove 'active' class from all tab buttons
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));

    // Remove 'active' class from all content sections
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    // Add 'active' class to the selected tab button
    const button = Array.from(document.querySelectorAll('.tab')).find(btn => btn.textContent.includes(tabIdToEmoji(tabId)));
    if (button) button.classList.add('active');

    // Add 'active' class to the selected content section
    const selectedContent = document.getElementById(tabId);
    if (selectedContent) selectedContent.classList.add('active');
}

function tabIdToEmoji(tabId) {
    const map = {
        practice: '🎯',
        dashboard: '📊',
        analytics: '📈',
        questions: '❓',
        settings: '⚙️'
    };
    return map[tabId] || '';
}

function getInitials(fullName) {
    if (!fullName) return "";
    return fullName
      .trim()
      .split(/\s+/)              // sépare sur les espaces multiples
      .filter(Boolean)           // retire les chaînes vides
      .map(word => word[0])      // prend la première lettre
      .join("")                  // concatène
      .slice(0, 2)               // max 2 lettres
      .toUpperCase();
  }

document.addEventListener("DOMContentLoaded", function () {

    const logoutBtn = document.getElementById('logoutBtn');
    const btnPractice = document.getElementById('btnPractice');
    if (logoutBtn) logoutBtn.onclick = () => {
        console.log("Déconnexion");
        localStorage.removeItem('authToken');
        window.location.href = 'login.html';
    };
    if (btnPractice) btnPractice.onclick = () => {
        console.log("btnPractice cliqué");
        window.location.href = 'test_interview_real_time.html';
    };
});


document.addEventListener("DOMContentLoaded", async () => {
    const token = localStorage.getItem("authToken"); // ou sessionStorage, selon ton choix
    if (!token) return console.error("Aucun token trouvé");

    try {
        const response = await fetch("/api/user-info", {
            method: "GET",
            headers: {
                Authorization: `Bearer ${token}`,
            },
        });

        if (!response.ok) throw new Error("Échec récupération user");

        const data = await response.json();

        // Met à jour l'affichage dynamique
        document.getElementById("user-name").textContent = data.full_name;
        document.getElementById("user-position").textContent = data.target_position;
        document.getElementById("session-count").textContent = data.session_count;
        // *** Initiales dynamiques ***
        document.getElementById("user-initials").textContent = getInitials(data.full_name);


    } catch (err) {
        console.error("Erreur récupération infos utilisateur :", err);
    }
});

document.addEventListener("DOMContentLoaded", async () => {
    const token = localStorage.getItem("authToken");
    if (!token) return console.error("Token manquant");

    try {
        const response = await fetch("/api/dashboard-metrics", {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error("Échec récupération métriques");

        const data = await response.json();

        // Met à jour chaque carte
        document.getElementById("metric-positive-value").textContent = data.positive_sentiment.value;
        document.getElementById("metric-positive-change").textContent = data.positive_sentiment.change;

        document.getElementById("metric-practice-value").textContent = data.practice_time.value;
        document.getElementById("metric-practice-change").textContent = data.practice_time.change;

        document.getElementById("metric-emotion-value").textContent = data.dominant_emotion.value;
        document.getElementById("metric-emotion-change").textContent = data.dominant_emotion.change;

        document.getElementById("metric-sessions-value").textContent = data.sessions.value;
        document.getElementById("metric-sessions-change").textContent = data.sessions.change;

    } catch (err) {
        console.error("Erreur dashboard :", err);
    }
});


document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token'); // ⚠️ Ton token JWT
    const questionList = document.getElementById('question-list');
    const addBtn = document.getElementById('add-question-btn');
    const form = document.getElementById('add-question-form');
    const submitBtn = document.getElementById('submit-question-btn');
    const newQuestionText = document.getElementById('new-question-text');

    // ➤ Charger les questions liées à l'utilisateur
    fetch('/questions/', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(res => res.json())
    .then(data => {
        data.forEach(question => {
            const item = document.createElement('div');
            item.classList.add('question-item');

            item.innerHTML = `
                <div class="question-text">${question.question_text}</div>
                <div class="question-meta">ID: ${question.id}</div>
            `;
            questionList.appendChild(item);
        });
    })
    .catch(err => {
        console.error('Erreur lors du chargement des questions :', err);
    });

    // ➤ Afficher le formulaire
    addBtn.addEventListener('click', () => {
        form.style.display = form.style.display === 'none' ? 'block' : 'none';
    });

    // ➤ Soumettre une nouvelle question
    submitBtn.addEventListener('click', () => {
        const questionText = newQuestionText.value.trim();
        if (!questionText) return alert("Merci d’écrire une question.");

        fetch('/add-question/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                question_text: questionText
            })
        })
        .then(res => {
            if (!res.ok) throw new Error("Échec de l'ajout");
            return res.json();
        })
        .then(newQuestion => {
            // Ajouter la question directement à la liste
            const item = document.createElement('div');
            item.classList.add('question-item');
            item.innerHTML = `
                <div class="question-text">${newQuestion.question_text}</div>
                <div class="question-meta">ID: ${newQuestion.id}</div>
            `;
            questionList.appendChild(item);

            newQuestionText.value = "";
            form.style.display = 'none';
        })
        .catch(err => {
            console.error("Erreur lors de l'ajout :", err);
            alert("Erreur lors de l'ajout de la question.");
        });
    });
});

*/

// ======================================
// 🔄 Onglets de navigation
// ======================================

function tabIdToEmoji(tabId) {
    const map = {
        practice: '🎯',
        dashboard: '📊',
        analytics: '📈',
        questions: '❓',
        settings: '⚙️'
    };
    return map[tabId] || '';
}

function switchTab(tabId) {
    // Supprime la classe 'active' de tous les onglets et contenus
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    // Ajoute la classe 'active' à l’onglet sélectionné
    const button = Array.from(document.querySelectorAll('.tab')).find(btn => btn.textContent.includes(tabIdToEmoji(tabId)));
    if (button) button.classList.add('active');

    // Affiche la section correspondante
    const selectedContent = document.getElementById(tabId);
    if (selectedContent) selectedContent.classList.add('active');
}

// ======================================
// 👤 Utilitaire : extraire les initiales
// ======================================

function getInitials(fullName) {
    if (!fullName) return "";
    return fullName
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .map(word => word[0])
        .join("")
        .slice(0, 2)
        .toUpperCase();
}

// ======================================
// 📋 Listeners généraux (boutons / navigation)
// ======================================

document.addEventListener("DOMContentLoaded", function () {
    const logoutBtn = document.getElementById('logoutBtn');
    const btnPractice = document.getElementById('btnPractice');

    if (logoutBtn) logoutBtn.onclick = () => {
        console.log("Déconnexion");
        localStorage.removeItem('authToken');
        window.location.href = 'login.html';
    };

    if (btnPractice) btnPractice.onclick = () => {
        console.log("btnPractice cliqué");
        window.location.href = 'test_interview_real_time.html';
    };
});

// ======================================
// 👤 Chargement des infos utilisateur
// ======================================

document.addEventListener("DOMContentLoaded", async () => {
    const token = localStorage.getItem("authToken");
    if (!token) return console.error("Aucun token trouvé");

    try {
        const response = await fetch("/api/user-info", {
            method: "GET",
            headers: {
                Authorization: `Bearer ${token}`,
            },
        });

        if (!response.ok) throw new Error("Échec récupération user");

        const data = await response.json();

        document.getElementById("user-name").textContent = data.full_name;
        document.getElementById("user-position").textContent = data.target_position;
        document.getElementById("session-count").textContent = data.session_count;
        document.getElementById("user-initials").textContent = getInitials(data.full_name);
    } catch (err) {
        console.error("Erreur récupération infos utilisateur :", err);
    }
});



document.addEventListener("DOMContentLoaded", async () => {
    const token = localStorage.getItem("authToken");
    if (!token) return console.error("Token manquant");

    try {
        // 🔹 Récupération des métriques principales
        const response = await fetch("/api/dashboard-metrics", {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error("Échec récupération métriques");

        const data = await response.json();

        document.getElementById("metric-positive-value").textContent = data.positive_sentiment.value;
        document.getElementById("metric-positive-change").textContent = data.positive_sentiment.change;

        document.getElementById("metric-practice-value").textContent = data.practice_time.value;
        document.getElementById("metric-practice-change").textContent = data.practice_time.change;

        document.getElementById("metric-emotion-value").textContent = data.dominant_emotion.value;
        document.getElementById("metric-emotion-change").textContent = data.dominant_emotion.change;

        document.getElementById("metric-sessions-value").textContent = data.sessions.value;
        document.getElementById("metric-sessions-change").textContent = data.sessions.change;

        // 🔹 Appel à l’API de stress
        const stressRes = await fetch("/stress_analysis/user/", {
            headers: {
                Authorization: `Bearer ${token}`
            }
        });

        if (!stressRes.ok) throw new Error("Échec récupération stress");

        const stressData = await stressRes.json();
        const stressValues = stressData
            .map(e => e.stress_global)
            .filter(v => v !== null);

        if (stressValues.length > 0) {
            const avg = stressValues.reduce((a, b) => a + b, 0) / stressValues.length;

            document.getElementById("metric-stress-value").textContent = avg.toFixed(2);

            const stressChangeEl = document.getElementById("metric-stress-change");
            stressChangeEl.textContent =
                avg > 0.7 ? "High stress" :
                avg > 0.4 ? "Moderate" :
                "Low";

            stressChangeEl.className = "metric-change " +
                (avg > 0.7 ? "negative" :
                 avg > 0.4 ? "neutral" : "positive");
        } else {
            document.getElementById("metric-stress-value").textContent = "N/A";
            document.getElementById("metric-stress-change").textContent = "No data";
        }

    } catch (err) {
        console.error("Erreur dashboard :", err);
    }
});




// ======================================
// ❓ Chargement et ajout de questions
// ======================================

document.addEventListener("DOMContentLoaded", () => {
    const token = localStorage.getItem('authToken');
    const questionList = document.getElementById('question-list');
    const addBtn = document.getElementById('add-question-btn');
    const form = document.getElementById('add-question-form');
    const submitBtn = document.getElementById('submit-question-btn');
    const newQuestionText = document.getElementById('new-question-text');

    // ➤ Charger les questions existantes
    fetch('/questions/', {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })

    .then(res => res.json())
    .then(data => {
        data.forEach(question => {
            const item = document.createElement('div');
            item.classList.add('question-item');
            item.innerHTML = `
                <div class="question-text">${question.question_text}</div>
                <div class="question-meta">ID: ${question.question_id}</div>
            `;
            questionList.appendChild(item);
            console.log(questionList);
        });
    })
    .catch(err => {
        console.error('Erreur lors du chargement des questions :', err);
    });

    // ➤ Afficher/Masquer le formulaire
    addBtn?.addEventListener('click', () => {
        form.style.display = form.style.display === 'none' ? 'block' : 'none';
    });

    // ➤ Ajouter une question
    submitBtn?.addEventListener('click', () => {
        const questionText = newQuestionText.value.trim();
        if (!questionText) return alert("Merci d’écrire une question.");

        fetch('/add-question/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                question_text: questionText
            })
        })
        .then(res => {
            if (!res.ok) throw new Error("Échec de l'ajout");
            return res.json();
        })
        .then(newQuestion => {
            const item = document.createElement('div');
            item.classList.add('question-item');
            item.innerHTML = `
                <div class="question-text">${newQuestion.question_text}</div>
                <div class="question-meta">ID: ${newQuestion.id}</div>
            `;
            questionList.appendChild(item);

            newQuestionText.value = "";
            form.style.display = 'none';
        })
        .catch(err => {
            console.error("Erreur lors de l'ajout :", err);
            alert("Erreur lors de l'ajout de la question.");
        });
    });
});
