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

  if (chatFeed) {
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  if (chatForm) {
    const step = parseInt(document.body.dataset.step || "0", 10);
    const totalSteps = parseInt(document.body.dataset.totalSteps || "0", 10);
    chatForm.addEventListener("submit", (event) => {
      if (step === totalSteps - 1) {
        event.preventDefault();
        if (searching) {
          searching.classList.add("active");
        }
        setTimeout(() => chatForm.submit(), 1200);
      }
    });
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
