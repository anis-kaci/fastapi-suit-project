/*async function fetchUserSessions(limit) {
    const resp = await fetch(`/api/user-sessions?limit=${limit}`);
    if (!resp.ok) throw new Error('Erreur chargement sessions');
    return await resp.json();
  }*/
console.log("dashboard.js chargé");
console.log("Token actuel :", localStorage.getItem("authToken"));
const token = localStorage.getItem('authToken');
async function fetchUserSessions(limit) {
  if (!token) {
    alert("Vous n'êtes pas connecté. Redirection vers la page de login.");
    window.location.href = 'login.html';
    return;
  }

  const resp = await fetch(`/api/user-sessions?limit=${limit}`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  if (!resp.ok) throw new Error('Erreur chargement sessions');
  return await resp.json();
}
    

function renderTable(sessions) {
  const tbody = document.getElementById('sessionsTableBody');
  tbody.innerHTML = '';
  sessions.forEach(s => {
    const tr = document.createElement('tr');
    ['session_id', 'created_at', 'dominant_emotion', 'dominant_sentiment_facial', 'dominant_sentiment_text', 'text_confidence']
      .forEach(key => {
        const td = document.createElement('td');
        td.textContent = s[key] ?? '-';
        tr.appendChild(td);
      });
    tbody.appendChild(tr);
  });
}

document.getElementById('refreshBtn').addEventListener('click', async () => {
  const limit = document.getElementById('sessionLimit').value;
  try {
    const data = await fetchUserSessions(limit);
    renderTable(data);
  } catch(e) {
    alert(e.message);
  }
});

window.onload = () => document.getElementById('refreshBtn').click();