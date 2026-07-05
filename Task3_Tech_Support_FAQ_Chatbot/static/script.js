const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const themeToggle = document.getElementById("theme-toggle");
const themeLabel = document.getElementById("theme-label");

// ---------- Theme handling ----------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeLabel.textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
applyTheme(prefersDark ? "dark" : "light");

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

// ---------- Message rendering ----------
function addMessage(text, sender) {
  const message = document.createElement("div");
  message.className = `message ${sender}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${sender === "bot" ? "bot-avatar" : "user-avatar"}`;
  avatar.textContent = sender === "bot" ? "TH" : "You";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  message.appendChild(avatar);
  message.appendChild(bubble);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

function showTyping() {
  const message = document.createElement("div");
  message.className = "message bot";
  message.id = "typing-message";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot-avatar";
  avatar.textContent = "TH";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<span class="typing-indicator"><span></span><span></span><span></span></span>`;

  message.appendChild(avatar);
  message.appendChild(bubble);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeTyping() {
  const typing = document.getElementById("typing-message");
  if (typing) typing.remove();
}

// ---------- Sending messages ----------
async function sendMessage(text) {
  addMessage(text, "user");
  chatInput.value = "";
  showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await response.json();
    removeTyping();
    addMessage(data.reply, "bot");
  } catch (err) {
    removeTyping();
    addMessage("Something went wrong reaching the server. Please try again.", "bot");
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  sendMessage(text);
});

// ---------- Topic chip shortcuts ----------
document.querySelectorAll(".topic-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const question = chip.getAttribute("data-q");
    sendMessage(question);
  });
});
