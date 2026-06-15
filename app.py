import streamlit as st
from deep_translator import GoogleTranslator

# --- Page Layout Configuration ---
st.set_page_config(page_title="AI Language Portal", page_icon="🌐", layout="wide")

st.title("🌐 AI Multi-Language Translator Portal")
st.markdown("An NLP application that automatically detects source phrasing and translates text across global dialects instantly.")

# Supported Languages Mapping
LANGUAGE_MAP = {
    "English": "en",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Urdu (اردو)": "ur",
    "Arabic (العربية)": "ar",
    "Chinese (中文)": "zh-CN",
    "Japanese (日本語)": "ja",
    "Italian (Italiano)": "it",
    "Russian (Русский)": "ru"
}

# --- Main Layout Split ---
col_input, col_output = st.columns(2)

with col_input:
    st.subheader("📥 Source Text")
    
    # Target Selection drop-down
    target_lang_label = st.selectbox(
        "Select Target Language:", 
        options=list(LANGUAGE_MAP.keys()), 
        index=1  # Defaults to Spanish
    )
    target_lang_code = LANGUAGE_MAP[target_lang_label]
    
    # Input box
    source_text = st.text_area(
        "Enter text to translate:", 
        value="Hello, welcome to our artificial intelligence laboratory task. I hope this application runs perfectly!",
        height=200
    )

# --- Processing Pipeline ---
if st.button("Translate Text", type="primary"):
    if not source_text.strip():
        st.warning("Please type or paste some text first!")
    else:
        try:
            # 1. Core Translation Request (Auto-detects the source language)
            translator = GoogleTranslator(source='auto', target=target_lang_code)
            translated_result = translator.translate(source_text)
            
            # Identify what language was auto-detected
            detected_lang = translator.source
            
            # 2. Reverse Verification Translation (For validation quality)
            reverse_translator = GoogleTranslator(source=target_lang_code, target='en')
            reverse_result = reverse_translator.translate(translated_result)

            # Display Outputs
            with col_output:
                st.subheader("📤 Translation Output")
                
                # Success banner displaying auto-detection details
                st.success(f"🤖 **Detected Source Language:** `{detected_lang.upper()}`")
                
                # Clean block container for target text
                st.text_area("Translated Text:", value=translated_result, height=120, disabled=True)
                
                # --- Text Metrics Expander ---
                with st.expander("📊 Text Analytics Metrics"):
                    c1, c2 = st.columns(2)
                    c1.metric("Character Count", len(source_text))
                    c2.metric("Word Count", len(source_text.split()))
                
                # --- Advanced NLP Verification Feature ---
                with st.expander("🔄 Accuracy Verification Check (Back to English)"):
                    st.caption("Translating your output back into English to check contextual accuracy:")
                    st.info(reverse_result)
                    
        except Exception as e:
            st.error(f"An error occurred during translation: {e}")
