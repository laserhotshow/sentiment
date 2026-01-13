import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import random
from nltk.sentiment import SentimentIntensityAnalyzer
import base64

# -------------------------------
# Background Setup
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.set_page_config(
        page_title="Customer Feedback Analyzer",
        page_icon="💬",
        layout="wide"
    )

    st.markdown(
        f"""
        <style>
        /* Full-page background */
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            z-index: -1;
        }}

        /* Text */
        h1,h2,h3,h4,p,label,span {{
            color: white !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.7);
        }}

        /* Buttons */
        button, div.stButton > button {{
            color: white !important;
            background-color: #1f77b4 !important;
            border: 2px solid white !important;
            font-weight: bold !important;
        }}
        button:hover, div.stButton > button:hover {{
            background-color: #0d4a7f !important;
        }}

        /* Slider */
        input[type="range"] {{
            background-color: #1f77b4 !important;
        }}

        /* Metrics */
        .stMetricValue, .stMetricLabel {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("b-1024x768-1.jpg")

# -------------------------------
# NLTK Sentiment
# -------------------------------
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# -------------------------------
# Refresh / Start Over Button
# -------------------------------
st.title("Customer Feedback Sentiment Analyzer")
st.write("Generate random themed reviews and analyze sentiment.")

if st.button("🔄 Refresh / Start Over"):
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]
    st.session_state.clear()

# -------------------------------
# Review Generator
# -------------------------------
st.subheader("Generate Sample CSVs by Theme")
theme = st.selectbox("Choose a theme", ["Weather","Service","Feedback","Reviews"])
num_reviews = st.slider("Number of reviews to generate", 5, 20, 5)

subjects = ["I","We","My friend","The team","Customer service","The product","The staff","Our experience"]
verbs_positive = ["loved","enjoyed","appreciated","valued","adored","found excellent"]
verbs_negative = ["hated","disliked","was disappointed by","was frustrated by","couldn't enjoy"]
verbs_neutral = ["experienced","noticed","found","tried","observed","encountered"]
objects_weather = ["the sunny day","the rain","stormy conditions","the cold weather","the heatwave"]
objects_service = ["the staff","the support team","the customer service","the assistance","the guidance"]
objects_feedback = ["the process","the feedback system","the form","the survey","the response"]
objects_reviews = ["the review process","the opinions shared","the ratings","the comments","the evaluations"]
adjectives = ["amazing","terrible","okay","fantastic","poor","average","excellent","mediocre","satisfying","disappointing"]
contexts = ["during a rainy afternoon","on a sunny morning","after a long wait","before the storm","while deciding","after reviewing all options"]

def generate_fluent_review(theme_name):
    subj = random.choice(subjects)
    if theme_name=="Weather":
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_weather)
    elif theme_name=="Service":
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_service)
    elif theme_name=="Feedback":
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_feedback)
    else:
        verb = random.choice(verbs_positive + verbs_negative + verbs_neutral)
        obj = random.choice(objects_reviews)
    adj = random.choice(adjectives)
    context = random.choice(contexts) if random.random() < 0.6 else ""
    templates = [
        f"{subj} {verb} {obj}. It was {adj} {context}.",
        f"Overall, {subj.lower()} {verb} {obj} and found it {adj}. {context}".strip(),
        f"{subj} felt that {obj} was {adj}. {context}",
        f"In my opinion, {obj} was {adj} {context}. {subj} {verb} it thoroughly.",
        f"{subj} would say the {obj} was {adj}. {context}."
    ]
    return random.choice(templates)

# Generate reviews
if st.button(f"Generate {num_reviews} Themed Reviews CSV"):
    reviews = [generate_fluent_review(theme) for _ in range(num_reviews)]
    df_generated = pd.DataFrame({"review": reviews})
    st.session_state.df_generated = df_generated
    st.success(f"{num_reviews} reviews generated!")
    st.dataframe(df_generated)

# -------------------------------
# Sentiment Analysis
# -------------------------------
if "df_generated" in st.session_state:
    df = st.session_state.df_generated

    def analyze_sentiment(text):
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "Positive", compound
        elif compound <= -0.05:
            return "Negative", compound
        else:
            return "Neutral", compound

    df[["sentiment","confidence"]] = df["review"].apply(lambda x: pd.Series(analyze_sentiment(str(x))))
    cols = [c for c in df.columns if c not in ["sentiment","confidence"]] + ["sentiment","confidence"]
    df = df[cols]

    total = len(df)
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    neu = (df["sentiment"] == "Neutral").sum()

    st.subheader("Sentiment Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Reviews", total)
    c2.metric("Positive", pos)
    c3.metric("Negative", neg)
    c4.metric("Neutral", neu)

    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment","Count"]
    sentiment_order = ["Positive","Neutral","Negative"]
    sentiment_counts["Sentiment"] = pd.Categorical(sentiment_counts["Sentiment"], categories=sentiment_order, ordered=True)
    sentiment_counts = sentiment_counts.sort_values("Sentiment")

    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        text="Count",
        color="Sentiment",
        color_discrete_map={"Positive":"#28a745","Negative":"#dc3545","Neutral":"#6c757d"},
        title="Customer Sentiment Distribution"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Feedback Analysis")
    st.dataframe(df, use_container_width=True, height=500)
    st.success("Analysis complete!")
