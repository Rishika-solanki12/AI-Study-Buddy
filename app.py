import os
import re
import json
import base64
import shutil
import subprocess
import sqlite3
import uuid
from datetime import datetime
import hashlib

import streamlit as st
import streamlit.components.v1 as components

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from gtts import gTTS
from groq import Groq
from huggingface_hub import InferenceClient
from ddgs import DDGS
import requests
from pathlib import Path

# ==========================================================
# HUGGING FACE SECRET SETUP
# ==========================================================

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide"
)

st.set_option(
    "client.toolbarMode",
    "viewer"
)

st.markdown("""
<style>
[data-baseweb="tab-list"] button[data-baseweb="tab"] div,
[data-baseweb="tab-list"] button[data-baseweb="tab"] span,
[data-baseweb="tab-list"] button[data-baseweb="tab"] p {
    font-size: 24px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
<style>
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

a[aria-label="App Creator Avatar"] {
    display: none !important;
}

.stButton>button {
    transition: all 0.3s ease;
    border-radius: 8px;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    border-color: #4CAF50;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}

html,
body {
    overflow-x: hidden !important;
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    overflow-x: hidden !important;
}

/* ==========================================================
   SIDEBAR WIDTH
   ========================================================== */

section[data-testid="stSidebar"] {
    min-width: 420px !important;
    max-width: 420px !important;
    width: 420px !important;
}

iframe {
    max-width: 100% !important;
    border: none !important;
}

[class*="viewerBadge"],
[class*="styles_viewerBadge"],
[data-testid="stAppDeployButton"] {
    display: none !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ==========================================================
# SESSION STATE
# ==========================================================

DEFAULT_STATE = {
    "messages": [],
    "processed_files": [],
    "vector_store": None,
    "generated_ai_image": None,
    "generated_ai_image_prompt": None,
    "real_image_search_enabled": True,
    "memory_user_id": None,
    "memory_loaded": False,
    "document_explanation": None,
    "image_explanation": None,
    "document_speaker_text": None,
    "document_speaker_text_language": None,
    "image_speaker_text": None,
    "image_speaker_text_language": None,
    "speaker_text": None,
    "speaker_text_language": None,
    "quiz_data": None,
    "flashcards": [],
    "mindmap_topics": [],
    "selected_mindmap_topic": None,
    "camera_enabled": False,
    "analyzed_image_name": None,
    "image_sentence_count": 3,
    "last_audio_id": None,
    # ======================================================
    # MAIN CHAT AUTO-OPEN
    # ======================================================
    "open_main_chat": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# API KEY CHECK
# ==========================================================

def get_groq_api_key():
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

    return str(api_key).strip()


# ==========================================================
# MODEL NAMES
# ==========================================================

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.6-27b"


# ==========================================================
# SAFE MODEL CREATION
# ==========================================================

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=get_groq_api_key(),
        model=TEXT_MODEL,
        temperature=0
    )


def get_vision_llm():
    return ChatGroq(
        api_key=get_groq_api_key(),
        model=VISION_MODEL,
        temperature=0
    )


# ==========================================================
# SAFE RESPONSE TEXT EXTRACTION
# ==========================================================

def response_to_text(response):
    if response is None:
        return ""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return " ".join(parts)

    return str(content)


# ==========================================================
# CLEAN MODEL THINKING
# ==========================================================

def remove_thinking(text):
    if not text:
        return ""

    text = str(text)

    # Remove <think>...</think>
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove unclosed <think>...</END>
    text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove common thinking/reasoning sections
    text = re.sub(
        r"(?is)(here'?s a thinking process:|self-correction/refinement.*|self-correction/verification.*|output generation.*)$",
        "",
        text
    )

    return text.strip()


# ==========================================================
# SAFE LLM CALL
# ==========================================================

def ask_llm(prompt):
    try:
        model = get_llm()
        response = model.invoke([HumanMessage(content=str(prompt))])
        answer = response_to_text(response)
        answer = remove_thinking(answer)

        if not answer.strip():
            raise RuntimeError("Groq returned an empty response.")

        return answer.strip()

    except Exception as e:
        raise RuntimeError(f"Groq AI request failed: {e}")


# ==========================================================
# LONG-TERM MEMORY SYSTEM
# ==========================================================

MEMORY_DB = "ai_study_buddy_memory.db"


# ==========================================================
# DATABASE INITIALIZATION
# ==========================================================

def init_memory_database():
    connection = sqlite3.connect(
        MEMORY_DB,
        check_same_thread=False
    )
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            memory TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            importance INTEGER DEFAULT 5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, memory)
        )
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user
        ON memories(user_id)
        """
    )

    connection.commit()
    connection.close()


init_memory_database()


# ==========================================================
# GET / CREATE USER ID
# ==========================================================

def get_memory_user_id():
    try:
        existing_id = st.query_params.get("memory_user_id")
    except Exception:
        existing_id = None

    if existing_id:
        st.session_state.memory_user_id = existing_id
        return existing_id

    if not st.session_state.get("memory_user_id"):
        new_id = str(uuid.uuid4())
        st.session_state.memory_user_id = new_id

        try:
            st.query_params["memory_user_id"] = new
