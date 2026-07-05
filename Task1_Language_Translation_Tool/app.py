from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from gtts import gTTS
import io

app = Flask(__name__)
CORS(app)

# Supported languages for gTTS
LANG_MAP = {
    'en': 'en',
    'ur': 'ur',
    'ar': 'ar',
    'fr': 'fr',
    'de': 'de',
    'es': 'es',
    'zh': 'zh',
    'ja': 'ja',
    'tr': 'tr',
    'hi': 'hi'
}

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    gtts_lang = LANG_MAP.get(lang, 'en')

    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype='audio/mpeg',
            as_attachment=False
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)