// ===== DOM ELEMENTS =====
const sourceText = document.getElementById('sourceText');
const outputText = document.getElementById('outputText');
const sourceLang = document.getElementById('sourceLang');
const targetLang = document.getElementById('targetLang');
const translateBtn = document.getElementById('translateBtn');
const swapBtn = document.getElementById('swapBtn');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const ttsBtn = document.getElementById('ttsBtn');
const charCount = document.getElementById('charCount');
const statusMsg = document.getElementById('statusMsg');

// ===== CHARACTER COUNTER =====
sourceText.addEventListener('input', () => {
    const length = sourceText.value.length;

    if (length > 500) {
        sourceText.value = sourceText.value.substring(0, 500);
    }

    charCount.textContent = sourceText.value.length;

    // Turn red when near limit
    charCount.style.color = length >= 450 ? '#f87171' : '#475569';
});

// ===== TRANSLATE FUNCTION =====
async function translateText() {
    const text = sourceText.value.trim();

    if (!text) {
        showStatus('⚠️ Please enter some text to translate.', 'error');
        return;
    }

    const from = sourceLang.value;
    const to = targetLang.value;

    if (from === to) {
        showStatus('⚠️ Source and target languages are the same!', 'error');
        return;
    }

    // Set loading state
    translateBtn.classList.add('loading');
    translateBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Translating...</span>';
    outputText.innerHTML = '<span class="placeholder-text">Fetching translation...</span>';
    showStatus('', '');

    try {
        const langPair = `${from}|${to}`;
        const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${langPair}&de=nooruleman2006oct@gmail.com&mt=1`;

        const response = await fetch(url);
        const data = await response.json();
        if (data.responseStatus === 200) {
            const translated = data.responseData.translatedText;
            outputText.textContent = translated;
            showStatus('✅ Translation successful!', 'success');
        } else {
            outputText.innerHTML = '<span class="placeholder-text">Translation failed. Try again.</span>';
            showStatus('❌ Could not translate. Check your connection.', 'error');
        }

    } catch (error) {
        outputText.innerHTML = '<span class="placeholder-text">Something went wrong.</span>';
        showStatus('❌ Network error. Please try again.', 'error');
    } finally {
        // Reset button
        translateBtn.classList.remove('loading');
        translateBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> <span>Translate</span>';
    }
}

// ===== SWAP LANGUAGES =====
swapBtn.addEventListener('click', () => {
    const tempLang = sourceLang.value;
    sourceLang.value = targetLang.value;
    targetLang.value = tempLang;

    // Also swap text if output has content
    const currentOutput = outputText.textContent;
    const hasOutput = !outputText.querySelector('.placeholder-text') && currentOutput.trim() !== '';

    if (hasOutput) {
        sourceText.value = currentOutput;
        charCount.textContent = sourceText.value.length;
        outputText.innerHTML = '<span class="placeholder-text">Translation will appear here...</span>';
    }
});

// ===== CLEAR BUTTON =====
clearBtn.addEventListener('click', () => {
    sourceText.value = '';
    charCount.textContent = '0';
    charCount.style.color = '#475569';
    outputText.innerHTML = '<span class="placeholder-text">Translation will appear here...</span>';
    showStatus('', '');
});

// ===== COPY BUTTON =====
copyBtn.addEventListener('click', () => {
    const text = outputText.textContent;
    const hasPlaceholder = outputText.querySelector('.placeholder-text');

    if (!text || hasPlaceholder) {
        showStatus('⚠️ Nothing to copy yet!', 'error');
        return;
    }

    navigator.clipboard.writeText(text).then(() => {
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
        showStatus('✅ Copied to clipboard!', 'success');

        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
        }, 2000);
    });
});

// ===== TEXT TO SPEECH =====
let isSpeaking = false;
let currentAudio = null;

ttsBtn.addEventListener('click', async () => {
    if (isSpeaking) {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }
        isSpeaking = false;
        ttsBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen';
        showStatus('', '');
        return;
    }

    const text = outputText.textContent;
    const hasPlaceholder = outputText.querySelector('.placeholder-text');

    if (!text || hasPlaceholder) {
        showStatus('⚠️ Nothing to speak yet!', 'error');
        return;
    }

    const lang = targetLang.value;

    isSpeaking = true;
    ttsBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop';
    showStatus('🔊 Speaking...', 'success');

    try {
        const response = await fetch('http://127.0.0.1:5000/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, lang: lang })
        });

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        currentAudio = new Audio(audioUrl);

        currentAudio.play();

        currentAudio.onended = () => {
            isSpeaking = false;
            ttsBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen';
            showStatus('', '');
        };

    } catch (error) {
        isSpeaking = false;
        ttsBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Listen';
        showStatus('❌ TTS failed. Is Flask running?', 'error');
    }
});
// ===== TRANSLATE BUTTON CLICK =====
translateBtn.addEventListener('click', translateText);

// ===== ENTER KEY SHORTCUT (Ctrl + Enter) =====
sourceText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        translateText();
    }
});

// ===== STATUS HELPER =====
function showStatus(message, type) {
    statusMsg.textContent = message;
    statusMsg.className = 'status-msg';
    if (type) statusMsg.classList.add(type);
}