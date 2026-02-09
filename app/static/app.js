(() => {
  const chatForm = document.getElementById("chat-form");
  const searching = document.getElementById("searching");
  const toggle = document.getElementById("toggle-less");
  const lessResults = document.getElementById("less-results");
  const actionField = document.getElementById("action-field");
  const skipBtn = document.getElementById("skip-btn");
  const chatFeed = document.getElementById("chat-feed");
  const resumeForm = document.getElementById("resume-form");
  const loading = document.getElementById("loading");
  const adToggle = document.getElementById("ad-toggle");

  if (adToggle) {
    const setAdHidden = (hidden) => {
      document.body.classList.toggle("ad-hidden", hidden);
      adToggle.setAttribute("aria-label", hidden ? "Afficher la publicité" : "Masquer la publicité");
      adToggle.textContent = hidden ? "▶" : "×";
      try {
        localStorage.setItem("adHidden", hidden ? "1" : "0");
      } catch {
        // Ignore storage errors.
      }
    };

    let hidden = false;
    try {
      hidden = localStorage.getItem("adHidden") === "1";
    } catch {
      hidden = false;
    }
    setAdHidden(hidden);

    adToggle.addEventListener("click", () => {
      setAdHidden(!document.body.classList.contains("ad-hidden"));
    });
  }

  if (chatFeed) {
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  if (chatForm) {
    const step = parseInt(document.body.dataset.step || "0", 10);
    const totalSteps = parseInt(document.body.dataset.totalSteps || "0", 10);
    const role = document.body.dataset.role || "none";
    chatForm.addEventListener("submit", (event) => {
      if (role !== "none" && step === totalSteps - 1) {
        event.preventDefault();
        if (searching) {
          searching.classList.add("active");
        }
        setTimeout(() => chatForm.submit(), 1200);
      }
    });

    const chatInput = chatForm.querySelector("textarea[name=\"query\"]");
    if (chatInput) {
      chatInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          chatForm.requestSubmit();
        }
      });
    }
  }

  if (skipBtn && actionField) {
    skipBtn.addEventListener("click", () => {
      actionField.value = "skip";
    });
  }

  if (toggle && lessResults) {
    toggle.addEventListener("click", () => {
      const open = lessResults.classList.toggle("open");
      toggle.textContent = open ? "Masquer les pistes proches" : "Explorer des pistes proches";
    });
  }

  if (resumeForm && loading) {
    resumeForm.addEventListener("submit", () => {
      loading.classList.add("active");
    });
  }
})();
