import streamlit as st
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download necessary NLTK data safely within the app
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")

download_nltk_resources()

# --- App Layout & Header ---
st.set_page_config(page_title="NLP Lab Workspace", page_icon="🤖", layout="wide")

st.title("🤖 NLP Multi-Task Lab Workspace")
st.markdown("Welcome! Input your text below to test different Natural Language Processing tasks in real-time.")

---

# --- Text Input Area ---
user_text = st.text_area(
    "Enter the text you want to analyze:",
    value="Artificial Intelligence is transforming the world. Streamlit makes it incredibly easy to build data applications quickly! However, debugging complex neural models can sometimes be challenging.",
    height=150
)

if st.button("Run Analysis", type="primary"):
    if user_text.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Create tabs for different NLP tasks
        tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "🏷️ POS Tagging & Tokens", "🔑 Key Metrics"])

        # --- TAB 1: Sentiment Analysis ---
        with tab1:
            st.subheader("Sentiment Analysis (via TextBlob)")
            blob = TextBlob(user_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Metric displays
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Polarity (Sentiment)", value=f"{polarity:.2f}", help="-1 is negative, 1 is positive")
            with col2:
                st.metric(label="Subjectivity", value=f"{subjectivity:.2f}", help="0 is objective, 1 is subjective")
            
            # Friendly sentiment verdict
            if polarity > 0.1:
                st.success("✨ **Overall Sentiment:** Positive")
            elif polarity < -0.1:
                st.error("📉 **Overall Sentiment:** Negative")
            else:
                st.info("😐 **Overall Sentiment:** Neutral")

        # --- TAB 2: Tokenization & POS Tagging ---
        with tab2:
            st.subheader("Tokenization & Part-of-Speech Tagging")
            tokens = word_tokenize(user_text)
            pos_tags = nltk.pos_tag(tokens)
            
            st.write(f"**Total Tokens (Words/Punctuation):** {len(tokens)}")
            
            # Format POS tags for clean display
            formatted_tags = [{"Token": token, "POS Tag": tag} for token, tag in pos_tags]
            st.dataframe(formatted_tags, use_container_width=True)

        # --- TAB 3: Text Statistics & Stopwords ---
        with tab3:
            st.subheader("Text Statistics")
            words = [word.lower() for word in word_tokenize(user_text) if word.isalnum()]
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word not in stop_words]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Total Words (No punctuation):** {len(words)}")
                st.markdown(f"**Words after removing Stopwords:** {len(filtered_words)}")
            
            with col2:
                st.markdown("**Most Common Keywords (excluding stopwords):**")
                word_counts = Counter(filtered_words)
                for word, count in word_counts.most_common(3):
                    st.write(f"- `{word}`: {count} times")
