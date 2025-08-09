import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- VISUALS: Purple & Red Theme + Logo ---
st.markdown(
    """
    <style>
    .main { background-color: #f7f3fa; }
    h1 { color: #800080; }
    .stButton > button {background-color:#ec2224; color:white;}
    </style>
    """,
    unsafe_allow_html=True
)

st.image("image.jpg", width=180)  # <-- Make sure image.jpg is in the same folder

# Load FAQs CSV (make sure faqs.csv is in the same folder)
faqs = pd.read_csv("faqs.csv")

# Vectorize FAQ questions
vectorizer = TfidfVectorizer().fit(faqs['Question'])
faq_vectors = vectorizer.transform(faqs['Question'])

# TITLE
st.title("Cleardeals Sales Inquiry Portal")
st.markdown("👋 **Welcome! Ask me any sales or Cleardeals-related question below.** I’ll try to match it to our team's FAQs.")

# TOP 5 MOST ASKED QUESTIONS
st.markdown("### 🔥 Top 5 Most Asked Questions:")
for idx in range(min(5, len(faqs))):
    question = faqs['Question'][idx]
    if st.button(question, key=f"top_q_{idx}"):
        st.write(f"**Q:** {faqs['Question'][idx]}")
        st.write(f"**A:** {faqs['Answer'][idx]}")

# Show all FAQs
if st.button("Show All FAQs"):
    for i, row in faqs.iterrows():
        st.write(f"**Q{i+1}:** {row['Question']}")
        st.write(f"**A:** {row['Answer']}\n")

# Main chatbot input
user_input = st.text_input("Ask me a question:")

if user_input:
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, faq_vectors).flatten()
    top = similarities.argmax()
    if similarities[top] > 0.3:
        st.write(f"**Q:** {faqs['Question'][top]}")
        st.write(f"**A:** {faqs['Answer'][top]}")
    else:
        st.write(
            "Hmm, it seems I don’t have this answer in my list yet. But you’re in good hands! Reach out to our Sales Manager for expert help:"
        )
        st.info(
            """
            **Mahavir Vaya**
            - *Sales Manager, Cleardeals*
            - 📞 9723992255
            - ✉️ vaya.mahavir@cleardeals.co.in
            """
        )
        # Fallback Logging (updated for pandas v2)
        if os.path.exists("unanswered.csv"):
            log_df = pd.read_csv("unanswered.csv")
            log_df = pd.concat(
                [log_df, pd.DataFrame({'Unanswered Question': [user_input]})],
                ignore_index=True
            )
            log_df.to_csv("unanswered.csv", index=False)
        else:
            pd.DataFrame({'Unanswered Question': [user_input]}).to_csv("unanswered.csv", index=False)
