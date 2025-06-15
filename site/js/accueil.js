document.getElementById('btn1').onclick = () => {
    window.location.href = 'suit-web-app-bis.html';
};
document.getElementById('btn2').onclick = () => {
    window.location.href = 'test_interview_analysis.html';
};
document.getElementById('btn3').onclick = () => {
    window.location.href = 'test_interview_real_time.html';
};
document.getElementById('logoutBtn').onclick = () => {
    localStorage.removeItem('authToken');
    window.location.href = 'login.html';
};