import os
from together import Together
from gtts import gTTS
import io
from tokens import token_size
import streamlit as st
def translate(text, lang):
    # Load the Together API key from the environment variables
    together_api_key = "a3361ce0fad4459875650576edc9d826ac8edacbcfa5b88304ef6f04a757042a"
    # Setting up the model
    client = Together(api_key=together_api_key)

    # Setting up the prompt
    prompt = f"Translate the following text to {lang}: {text}"

    messages = [
        {
            "role": "system",
            "content": "You are a translator bot. Provide only the translated text."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    max_tokens = 8192 - token_size(prompt)  # Adjust max_tokens to fit within the limit

    try:
        # Get response
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.3,
            stop=["<|eot_id|>"],
            stream=True
        )

        response_text = ""
        for chunk in response:
            for choice in chunk.choices:
                if choice.text:
                    response_text += choice.text

        if not response_text.strip():
            raise ValueError("The translation response is empty.")

        return response_text

    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return "Translation error."

def generate_audio(text, lang):
    if not text:
        raise ValueError("No text to speak.")
    languages = {"English": "en", "French": "fr", "Spanish": "es"}
    lang_code = languages.get(lang, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io
