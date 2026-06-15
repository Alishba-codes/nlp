import streamlit as st
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Ensure NLTK resources are downloaded
@st.cache_resource
def load_nltk():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        st.error(f"Error loading NLTK: {e}")

load_nltk()

# --- Page Config ---
st.set_page_config(page_title="Enterprise Review Auditor", page_icon="📈", layout="wide")

st.title("📈 Enterprise Customer Feedback & Review Auditor")
st.markdown("Analyze customer reviews at scale, track sentiment metrics, and flag high-priority issues instantly.")

# --- Sidebar for Preset Demo Data ---
st.sidebar.header("📋 Load Demo Dataset")
demo_industry = st.sidebar.selectbox(
    "Choose an industry to test preset reviews:",
    ["Tech Gadget Reviews", "Hotel & Hospitality Feedback"]
)

# Preset data matching real-world scenarios
tech_reviews = [
    "The battery life on this laptop is absolutely incredible! Lasts a full 12 hours. Highly recommend.",
    "Worst customer service ever. The screen arrived cracked, and support refused to issue a refund.",
    "It works okay, but the keyboard feels a bit cheap for this price point. Mediocre experience.",
    "Extremely fast processor and beautiful display! Best purchase I have made this year.",
    "The software keeps crashing every time I try to open a heavy project. Very frustrating product."
]

hotel_reviews = [
    "The room was pristine, and the staff went above and beyond to make our anniversary special!",
    "Terrible experience. The AC didn't work all night, and the room smelled like damp smoke.",
    "Good location close to downtown, but the breakfast options were very limited and cold.",
    "Amazing rooftop view and excellent room service. Will definitely stay here again next time.",
    "Checking in took almost an hour. The front desk was understaffed and quite disorganized."
]

selected_reviews = tech_reviews if demo_industry == "Tech Gadget Reviews" else hotel_reviews

# --- Main Interface ---
st.subheader("📥 Input Reviews for Auditing")
user_input = st.text_area(
    "Paste your customer reviews here (One review per line):",
    value="\n".join(selected_reviews),
    height=180
)

# Process Data
if st.button("Analyze Dataset", type="primary"):
    reviews_list = [line.strip() for line in user_input.split("\n") if line.strip()]
    
    if not reviews_list:
        st.warning("Please enter at least one review to analyze.")
    else:
        # Data Pipeline Processing
        processed_data = []
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        all_cleaned_words = []
        stop_words = set(stopwords.words('english'))

        for review in reviews_list:
            blob = TextBlob(review)
            polarity = blob.sentiment.polarity
            
            # Classify Sentiment
            if polarity > 0.15:
                status = "Positive"
                sentiment_counts["Positive"] += 1
            elif polarity < -0.15:
                status = "Negative"
                sentiment_counts["Negative"] += 1
            else:
                status = "Neutral"
                sentiment_counts["Neutral"] += 1
            
            # Real-world logic: Flag negative reviews for immediate customer support action
            action_flag = "🚨 High Priority Action" if polarity < -0.2 else "✅ Normal"
            
            processed_data.append({
                "Review Text": review,
                "Sentiment Score": round(polarity, 2),
                "Category": status,
                "Audit Status": action_flag
            })

            # Tokenize for keyword analytics
            words = [w.lower() for w in word_tokenize(review) if w.isalnum()]
            all_cleaned_words.extend([w for w in words if w not in stop_words])

        # Convert to Pandas DataFrame for advanced manipulation
        df = pd.DataFrame(processed_data)

        # --- Dashboard Metrics Layout ---
        st.markdown("---")
        st.subheader("📊 Executive Summary Metrics")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Reviews Audited", len(df))
        m2.metric("Positive Reviews", sentiment_counts["Positive"], delta=f"{int(sentiment_counts['Positive']/len(df)*100)}%")
        m3.metric("Negative Reviews", sentiment_counts["Negative"], delta=f"-{int(sentiment_counts['Negative']/len(df)*100)}%", delta_color="inverse")
        m4.metric("Urgent Support Tickets", len(df[df["Audit Status"] == "🚨 High Priority Action"]))

        # --- Visualizations & Analytics ---
        col_chart, col_words = st.columns([1, 1])
        
        with col_chart:
            st.markdown("### 🥧 Sentiment Distribution")
            chart_data = pd.DataFrame({
                "Category": sentiment_counts.keys(),
                "Count": sentiment_counts.values()
            })
            # Interactive Streamlit Bar Chart
            st.bar_chart(chart_data.set_index("Category"), color="#29b5e8")

        with col_words:
            st.markdown("### 🔑 Trending Keywords")
            word_counts = Counter(all_cleaned_words)
            top_words = word_counts.most_common(5)
            
            if top_words:
                for word, count in top_words:
                    st.markdown(f"- **{word.title()}** — mentioned `{count}` times")
            else:
                st.write("Not enough text data to extract keywords.")

        # --- Data Table and Export ---
        st.markdown("---")
        st.subheader("📋 Audited Data Log")
        st.dataframe(df, use_container_width=True)

        # Download Report Feature
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Audit Report (CSV)",
            data=csv,
            file_name="customer_audit_report.csv",
            mime="text/csv"
        )
