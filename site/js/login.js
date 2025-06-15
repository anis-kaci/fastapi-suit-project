const form = document.getElementById('loginForm');
        const errorMsg = document.getElementById('errorMsg');
        const registerBtn = document.getElementById('registerBtn');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            errorMsg.textContent = "";

            const data = {
                email: form.email.value.trim(),
                password: form.password.value,
            };

            try {
                const response = await fetch('http://localhost:8000/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (response.ok) {
                    const resData = await response.json();
                    // Stocker le token JWT dans localStorage
                    localStorage.setItem('authToken', resData.token);
                    localStorage.setItem('sessionId', resData.session_id);
                    // Redirection vers la page d'accueil
                    window.location.href = 'index.html';
                } else {
                    const errorData = await response.json();
                    errorMsg.textContent = errorData.detail || 'Email ou mot de passe incorrect.';
                }
            } catch (error) {
                errorMsg.textContent = "Erreur réseau, veuillez réessayer plus tard.";
                console.error(error);
            }
        });

        registerBtn.addEventListener('click', () => {
            window.location.href = 'register.html';
        });