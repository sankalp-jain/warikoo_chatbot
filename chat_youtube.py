import os
import base64
import pandas as pd
import streamlit as st
from chat_functions import *

GROQ_API_KEY = st.secrets("GROQ_API_KEY")
YOUTUBE_API_KEY = st.secrets("YOUTUBE_API_KEY")
video_titles = pd.read_csv("Video Titles.csv")["Video Titles"].to_list()

st.markdown("""
    <style>
        .main {
            background-color: #000000;
            color: #ffffff;
        }
        .video-summary {
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        }
        .title {
            color: #FFD700;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .summary {
            color: #d3d3d3;
            font-size: 16px;
            line-height: 1.5;
        }
        .progress {
            color: #32CD32;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def get_image_base64(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
    return base64.b64encode(data).decode()

image_base64 = get_image_base64("warikoo.png")

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{image_base64}" style="width: 50px; margin-right: 10px;">
        <span style="font-size: 20px; font-weight: bold;">Warikoo Chatbot</span>
    </div>
    """,
    unsafe_allow_html=True
)

user_query = st.text_input("Discover insights from Warikoo's videos with ease! ✨", placeholder="Type something like 'how to be invest'...")

if user_query:
    st.markdown("### 🔍 Search Results Loading...")
    results = search_vector_store(user_query, video_titles)
    
    for i, result in enumerate(results, start=1):
        video_id, summary = summarize_youtube_video(YOUTUBE_API_KEY, result, GROQ_API_KEY)
        
        st.markdown(f"""
            <div class="video-summary">
                <div class="title">Video</div>
                <iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>
                <div class="summary">Summary: {summary}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("---")