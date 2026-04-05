/**
 * Spam Detector — Client-Side Logic
 * Handles form submission, API calls, and result rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    const form          = document.getElementById("predict-form");
    const messageInput  = document.getElementById("message-input");
    const analyzeBtn    = document.getElementById("analyze-btn");
    const clearBtn      = document.getElementById("clear-btn");
    const resultSection = document.getElementById("result-section");
    const resultBadge   = document.getElementById("result-badge");
    const resultEmoji   = document.getElementById("result-emoji");
    const resultLabel   = document.getElementById("result-label");
    const confidenceFill= document.getElementById("confidence-fill");
    const confidenceVal = document.getElementById("confidence-value");
    const hamProbVal    = document.getElementById("ham-prob-value");
    const spamProbVal   = document.getElementById("spam-prob-value");

    // --- Load model stats on page load ---
    fetchStats();

    // --- Form submission ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        setLoading(true);

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });

            if (!res.ok) throw new Error("Prediction failed");
            const data = await res.json();
            displayResult(data);
        } catch (err) {
            console.error(err);
            alert("Something went wrong. Please make sure the server is running.");
        } finally {
            setLoading(false);
        }
    });

    // --- Clear button ---
    clearBtn.addEventListener("click", () => {
        messageInput.value = "";
        resultSection.classList.add("hidden");
        messageInput.focus();
    });

    // --- Example buttons ---
    document.querySelectorAll(".example-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            messageInput.value = btn.dataset.message;
            messageInput.focus();
            // Auto-submit
            form.dispatchEvent(new Event("submit"));
        });
    });

    // --- Display prediction result ---
    function displayResult(data) {
        const isSpam = data.label === "spam";

        // Show result card
        resultSection.classList.remove("hidden");

        // Badge
        resultBadge.className = "result-badge " + data.label;
        resultEmoji.textContent = isSpam ? "🚫" : "✅";
        resultLabel.textContent = isSpam ? "SPAM DETECTED" : "LEGITIMATE MESSAGE";

        // Confidence bar
        confidenceFill.className = "confidence-bar-fill " + data.label;
        // Trigger reflow for animation
        confidenceFill.style.width = "0%";
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                confidenceFill.style.width = data.confidence + "%";
            });
        });
        confidenceVal.textContent = data.confidence + "%";

        // Probabilities
        hamProbVal.textContent  = data.ham_probability + "%";
        spamProbVal.textContent = data.spam_probability + "%";

        // Scroll into view
        resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // --- Fetch model stats ---
    async function fetchStats() {
        try {
            const res = await fetch("/stats");
            if (!res.ok) return;
            const data = await res.json();

            document.getElementById("stat-accuracy").textContent  = data.accuracy + "%";
            document.getElementById("stat-precision").textContent = data.precision + "%";
            document.getElementById("stat-recall").textContent    = data.recall + "%";
            document.getElementById("stat-f1").textContent        = data.f1_score + "%";

            document.getElementById("info-total").textContent = data.total_samples + " total";
            document.getElementById("info-ham").textContent   = data.ham_count + " ham";
            document.getElementById("info-spam").textContent  = data.spam_count + " spam";
        } catch (err) {
            console.log("Stats not loaded yet (server may still be starting).");
        }
    }

    // --- Loading state ---
    function setLoading(loading) {
        if (loading) {
            analyzeBtn.classList.add("loading");
            analyzeBtn.disabled = true;
        } else {
            analyzeBtn.classList.remove("loading");
            analyzeBtn.disabled = false;
        }
    }
});
