async function checkSession() {
    const token = localStorage.getItem('authToken');
    if (!token) {
        window.location.href = 'login.html';
        return;
    }

    try {
        const response = await fetch('http://localhost:8000/home', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        if (response.status === 401) {
            window.location.href = 'login.html';
            return;
        }
        if (!response.ok) {
            throw new Error("Erreur serveur");
        }
        const data = await response.json();
        console.log("Bienvenue " + data.user_name);
        // Tu peux aussi afficher un message personnalisé ici
    } catch (e) {
        console.error(e);
        alert("Erreur lors de la vérification de la session. Veuillez vous reconnecter.");
        window.location.href = 'login.html';
    }
}