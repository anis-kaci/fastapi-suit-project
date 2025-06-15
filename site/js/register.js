document.getElementById("registerForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const data = {
        first_name: document.getElementById("firstName").value,
        last_name: document.getElementById("lastName").value,
        birth_date: document.getElementById("birthDate").value,
        education_level: document.getElementById("educationLevel").value,
        target_position: document.getElementById("targetPosition").value,
        email: document.getElementById("email").value,
        password: document.getElementById("password").value
    };

    const response = await fetch("http://localhost:8000/register", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const errorBox = document.getElementById("errorMessage");
    const successBox = document.getElementById("successMessage");
    errorBox.textContent = "";
    successBox.textContent = "";

    if (response.ok) {
        successBox.textContent = "Compte créé avec succès ! Redirection...";
        setTimeout(() => {
            window.location.href = "login.html";
        }, 1500);
    } else {
        const err = await response.json();
        errorBox.textContent = err.detail || "Erreur lors de la création du compte.";
    }
});

document.getElementById("goToLogin").addEventListener("click", function() {
    window.location.href = "login.html";
});