/*console.log("accueil.js chargé");

document.getElementById('btn1').onclick = () => {
    console.log("btn1 cliqué");
    window.location.href = 'suit-web-app-bis.html';
};
document.getElementById('btn2').onclick = () => {
    console.log("btn2 cliqué");
    window.location.href = 'test_interview_analysis.html';
};
document.getElementById('btn3').onclick = () => {
    console.log("btn3 cliqué");
    window.location.href = 'test_interview_real_time.html';
};
document.getElementById('btsDash').onclick =()=> {
    console.log("le boutton dash est cliqué")
    window.location.href = 'dashboard.html'
}
document.getElementById('logoutBtn').onclick = () => {
    localStorage.removeItem('authToken');
    window.location.href = 'login.html';
};
*/

console.log("accueil.js chargé");

document.addEventListener("DOMContentLoaded", function () {
    const btn1 = document.getElementById('btn1');
    const btn2 = document.getElementById('btn2');
    const btn3 = document.getElementById('btn3');
    const btnDash = document.getElementById('btnDash'); // corrigé ici
    const logoutBtn = document.getElementById('logoutBtn');

    if (btn1) btn1.onclick = () => {
        console.log("btn1 cliqué");
        window.location.href = 'suit-web-app-bis.html';
    };

    if (btn2) btn2.onclick = () => {
        console.log("btn2 cliqué");
        window.location.href = 'test_interview_analysis.html';
    };

    if (btn3) btn3.onclick = () => {
        console.log("btn3 cliqué");
        window.location.href = 'test_interview_real_time.html';
    };

    if (btnDash) btnDash.onclick = () => {
        console.log("le bouton dash est cliqué");
        window.location.href = 'user_sessions.html';
    };

    if (logoutBtn) logoutBtn.onclick = () => {
        console.log("Déconnexion");
        localStorage.removeItem('authToken');
        window.location.href = 'login.html';
    };
});
