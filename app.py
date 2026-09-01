import os
import re
import json
import base64
import shutil
import subprocess
from datetime import datetime

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
import sqlite3
import uuid



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


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
<style>

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

    # ======================================================
    # LONG-TERM MEMORY
    # ======================================================
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
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value
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
#
# The ID is kept in the browser URL so that Streamlit
# reruns do not create a new memory identity.
# ==========================================================

def get_memory_user_id():

    try:

        existing_id = st.query_params.get(
            "memory_user_id"
        )

    except Exception:

        existing_id = None


    if existing_id:

        st.session_state.memory_user_id = (
            existing_id
        )

        return existing_id


    if not st.session_state.get(
        "memory_user_id"
    ):

        new_id = str(
            uuid.uuid4()
        )

        st.session_state.memory_user_id = (
            new_id
        )

        try:

            st.query_params["memory_user_id"] = (
                new_id
            )

        except Exception:

            pass


    return st.session_state.memory_user_id


memory_user_id = get_memory_user_id()


# ==========================================================
# SAVE MEMORY
# ==========================================================

def save_memory(
    memory_text,
    category="general",
    importance=5
):

    if not memory_text:
        return

    memory_text = str(
        memory_text
    ).strip()

    if not memory_text:
        return


    connection = sqlite3.connect(
        MEMORY_DB,
        check_same_thread=False
    )

    cursor = connection.cursor()

    now = datetime.now().isoformat()


    try:

        cursor.execute(
            """
            INSERT INTO memories
            (
                user_id,
                memory,
                category,
                importance,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT(user_id, memory)
            DO UPDATE SET
                category = excluded.category,
                importance = excluded.importance,
                updated_at = excluded.updated_at
            """,
            (
                memory_user_id,
                memory_text,
                category,
                int(importance),
                now,
                now
            )
        )

        connection.commit()

    except Exception:

        connection.rollback()

    finally:

        connection.close()


# ==========================================================
# LOAD ALL USER MEMORIES
# ==========================================================

def load_all_memories():

    connection = sqlite3.connect(
        MEMORY_DB,
        check_same_thread=False
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            memory,
            category,
            importance
        FROM memories
        WHERE user_id = ?
        ORDER BY
            importance DESC,
            updated_at DESC
        """,
        (
            memory_user_id,
        )
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# ==========================================================
# DELETE ALL MEMORIES
# ==========================================================

def delete_all_memories():

    connection = sqlite3.connect(
        MEMORY_DB,
        check_same_thread=False
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM memories
        WHERE user_id = ?
        """,
        (
            memory_user_id,
        )
    )

    connection.commit()

    connection.close()


# ==========================================================
# MEMORY SEARCH
#
# We do lightweight keyword matching so every question
# does NOT require another embedding/vector database.
# ==========================================================

def search_memories(
    query,
    max_memories=8
):

    memories = load_all_memories()

    if not memories:
        return []


    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9\u0900-\u097F]{3,}\b",
            str(query).lower()
        )
    )


    scored_memories = []


    for memory, category, importance in memories:

        memory_words = set(
            re.findall(
                r"\b[a-zA-Z0-9\u0900-\u097F]{3,}\b",
                str(memory).lower()
            )
        )


        overlap = len(
            query_words.intersection(
                memory_words
            )
        )


        score = (
            overlap * 10
            +
            int(importance)
        )


        scored_memories.append(
            (
                score,
                memory,
                category
            )
        )


    scored_memories.sort(
        key=lambda x: x[0],
        reverse=True
    )


    # Always include highly important memories.
    important_memories = [
        item
        for item in scored_memories
        if item[0] >= 5
    ]


    selected = []


    for item in important_memories:

        if len(selected) >= max_memories:
            break

        selected.append(item)


    # If there are not enough matching memories,
    # include recent/high-importance memories.
    if len(selected) < max_memories:

        for item in scored_memories:

            if item not in selected:

                selected.append(item)

            if len(selected) >= max_memories:
                break


    return [
        {
            "memory": item[1],
            "category": item[2]
        }
        for item in selected
    ]


# ==========================================================
# FORMAT MEMORY FOR THE MODEL
# ==========================================================

def get_memory_context(
    query,
    max_memories=8
):

    memories = search_memories(
        query,
        max_memories=max_memories
    )


    if not memories:

        return (
            "No long-term memory is available "
            "for this user."
        )


    lines = []

    for item in memories:

        lines.append(
            f"- {item['memory']}"
        )


    return "\n".join(
        lines
    )


# ==========================================================
# EXTRACT IMPORTANT USER MEMORIES
#
# The model decides what is worth remembering.
# It does NOT store every single message.
# ==========================================================

def extract_and_save_memories(
    user_message,
    assistant_message
):

    if not user_message:
        return


    try:

        memory_prompt = f"""
You are a long-term memory manager for an AI Study Buddy.

Your job is to identify ONLY useful, non-sensitive,
long-term information about the user that would help
the assistant provide better responses in future chats.

USER MESSAGE:

{user_message}

ASSISTANT RESPONSE:

{assistant_message}

SAVE information such as:

- User's preferred name
- Learning goals
- Subjects they are studying
- Programming languages they are learning
- Long-term projects
- Stable preferences
- Preferred explanation style
- Preferred language
- Repeated study interests
- Useful non-sensitive background information

DO NOT save:

- Passwords
- API keys
- OTPs
- Credit card information
- Bank information
- Exact addresses
- Private secrets
- Highly sensitive personal information
- Temporary information that is not useful later
- The complete conversation
- The assistant's response as a memory

IMPORTANT:

Only return memories that are genuinely useful later.

Return ONLY valid JSON.

Format:

[
  {{
    "memory": "Short useful memory",
    "category": "name|preference|goal|study|project|general",
    "importance": 1
  }}
]

Importance:

1-3 = low
4-6 = useful
7-8 = important
9-10 = very important

If there is nothing useful to remember, return:

[]

Do not write anything outside JSON.
"""


        memory_llm = ChatGroq(
            api_key=st.secrets[
                "GROQ_API_KEY"
            ],
            model_name="openai/gpt-oss-20b",
            temperature=0
        )


        response = memory_llm.invoke(
            [
                HumanMessage(
                    content=memory_prompt
                )
            ]
        )


        raw_memory = getattr(
            response,
            "content",
            ""
        )


        raw_memory = remove_thinking(
            str(raw_memory)
        ).strip()


        raw_memory = re.sub(
            r"```json",
            "",
            raw_memory,
            flags=re.IGNORECASE
        )


        raw_memory = raw_memory.replace(
            "```",
            ""
        ).strip()


        start = raw_memory.find("[")

        end = raw_memory.rfind("]")


        if (
            start == -1
            or
            end == -1
            or
            end <= start
        ):

            return


        raw_memory = raw_memory[
            start:end + 1
        ]


        memories = json.loads(
            raw_memory
        )


        if not isinstance(
            memories,
            list
        ):

            return


        for item in memories:

            if not isinstance(
                item,
                dict
            ):

                continue


            memory_text = str(
                item.get(
                    "memory",
                    ""
                )
            ).strip()


            category = str(
                item.get(
                    "category",
                    "general"
                )
            ).strip()


            try:

                importance = int(
                    item.get(
                        "importance",
                        5
                    )
                )

            except Exception:

                importance = 5


            importance = max(
                1,
                min(
                    importance,
                    10
                )
            )


            if (
                memory_text
                and
                len(memory_text) <= 500
            ):

                save_memory(
                    memory_text,
                    category,
                    importance
                )


    except Exception:

        # Memory must NEVER break the main chatbot.
        pass


# ==========================================================
# MEMORY SIDEBAR
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🧠 Long-Term Memory"
)


current_memories = load_all_memories()


if current_memories:

    st.sidebar.success(
        f"🧠 {len(current_memories)} "
        f"memory item(s) saved"
    )

else:

    st.sidebar.info(
        "No long-term memories saved yet."
    )


if st.sidebar.button(
    "🗑️ Forget My Long-Term Memory",
    key="forget_long_term_memory"
):

    delete_all_memories()

    st.sidebar.success(
        "✅ Long-term memory deleted."
    )

    st.rerun()



# ==========================================================
# LANGUAGE SETTINGS
# ==========================================================

EXPLANATION_LANGUAGES = [
    "English",
    "Hindi",
    "Hinglish",
    "Bengali",
    "Marathi",
    "Tamil",
    "Telugu",
    "Gujarati",
    "Punjabi",
    "French",
    "Spanish"
]


SPEAKER_LANGUAGE_CODES = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "French": "fr-FR",
    "Spanish": "es-ES"
}


TTS_LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "French": "fr",
    "Spanish": "es"
}


# ==========================================================
# HELPERS
# ==========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="openai/gpt-oss-20b",
        temperature=0
    )


llm = get_llm()


groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


def ask_llm(prompt):

    response = llm.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    return response.content


# ==========================================================
# CLEAN MODEL THINKING
# ==========================================================

def remove_thinking(text):

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return text.strip()


# ==========================================================
# JSON CLEANER
# ==========================================================

def clean_json_response(text):

    if not text:
        return ""

    text = remove_thinking(text)

    text = str(text).strip()

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    first = text.find("[")

    if first != -1:
        text = text[first:]

    last = text.rfind("]")

    if last != -1:
        text = text[:last + 1]

    return text.strip()


# ==========================================================
# SPEECH CLEANER
# ==========================================================

def clean_text_for_speech(text):

    if not text:
        return ""

    text = str(text)

    text = remove_thinking(text)

    text = re.sub(
        r"```.*?```",
        " ",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"(?m)^\s*#{1,6}\s*",
        "",
        text
    )

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = text.replace("~~", "")

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    bullet_symbols = [
        "•", "●", "○", "▪", "▫", "◦",
        "►", "▸", "▹", "◆", "◇",
        "✓", "✔", "☑", "☛",
        "➜", "➤"
    ]

    for symbol in bullet_symbols:
        text = text.replace(symbol, " ")

    arrow_symbols = [
        "→", "⇒", "⟶", "⟹",
        "←", "↔", "↕", "↑",
        "↓", "➝", "➞", "➜", "➡"
    ]

    for symbol in arrow_symbols:
        text = text.replace(symbol, " ")

    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("|", " ")

    unwanted_symbols = [
        "§", "©", "®", "™", "※",
        "★", "☆", "✦", "✧", "✱",
        "✳", "❖", "♥", "♦", "♣",
        "♠", "∞", "≈", "≠", "≤",
        "≥", "±", "÷", "×", "√",
        "∑", "∫", "∂", "∆", "∇",
        "~", "^", "_", "=", "@"
    ]

    for symbol in unwanted_symbols:
        text = text.replace(symbol, " ")

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub(
        " ",
        text
    )

    text = re.sub(
        r"(?m)^\s*(\d+)\s*[\)\-:]\s*",
        r"\1. ",
        text
    )

    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    text = re.sub(
        r"!{2,}",
        "!",
        text
    )

    text = re.sub(
        r"\?{2,}",
        "?",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n",
        text
    )

    return text.strip()


# ==========================================================
# LANGUAGE-SPECIFIC INSTRUCTIONS
# ==========================================================

def get_language_instruction(language):

    if language == "English":

        return """
Write the complete answer in natural English.
Use English alphabet only.
Do not mix Hindi or other languages.
"""

    elif language == "Hindi":

        return """
Write the COMPLETE answer in pure Hindi.

VERY IMPORTANT:
- Use Devanagari script.
- Use Hindi words wherever possible.
- Do NOT write Hindi using English/Roman letters.
- Do NOT use Hinglish.
- Do NOT mix English sentences into the answer.
- Technical terms may remain in English only when absolutely necessary.
- The actual explanation must be predominantly Devanagari Hindi.
- Return ONLY the Hindi answer.
"""

    elif language == "Hinglish":

        return """
Write the complete answer in natural Hinglish.
Use a comfortable mixture of Hindi and English.
Hindi may be written in Roman/English script.
Do not make it unnecessarily formal.
"""

    else:

        return f"""
Write the complete answer in {language}.
Use the correct natural writing system for that language.
Do not mix languages unnecessarily.
Return ONLY the requested language.
"""


# ==========================================================
# TRANSLATION HELPER
# ==========================================================

def translate_for_speech(
    text,
    target_language
):

    if not text:
        return ""

    cleaned_source = remove_thinking(text)

    language_rules = get_language_instruction(
        target_language
    )

    prompt = f"""
You are a professional translator.

Translate the following educational explanation
into {target_language}.

{language_rules}

STRICT RULES:

1. Translate ONLY the actual explanation.
2. Do not add greetings.
3. Do not add introduction.
4. Do not mention AI.
5. Do not add extra information.
6. Do not remove important information.
7. Preserve the original meaning exactly.
8. Keep approximately the same paragraph and sentence structure.
9. Do not add explanations about the translation.
10. Do not write "Here is the translation".
11. Return ONLY the translated text.

SOURCE TEXT:

{cleaned_source}
"""

    translator = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="qwen/qwen3.6-27b",
        temperature=0
    )

    response = translator.invoke(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )

    translated = getattr(
        response,
        "content",
        ""
    )

    translated = remove_thinking(
        translated
    )

    translated = clean_text_for_speech(
        translated
    )

    return translated


# ==========================================================
# LANGUAGE SELECTORS
#
# IMPORTANT:
# These are intentionally placed BEFORE file processing.
# This fixes:
# "name 'translation_language' is not defined"
# ==========================================================

# ==========================================================
# TITLE
# ==========================================================

st.title("📚 AI Study Buddy")

st.write(
    "Upload your study material and search concepts instantly!"
)


# ==========================================================
# SIDEBAR UPLOAD
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.header(
    "📁 Upload Study Material"
)


uploaded_files = st.sidebar.file_uploader(
    "Upload Study Material",
    type=[
        "pdf",
        "doc",
        "docx",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "gif",
        "bmp",
        "tiff",
        "tif",
        "heic",
        "heif"
    ],
    accept_multiple_files=True
)


# ==========================================================
# CAMERA
# ==========================================================

st.sidebar.subheader(
    "📷 Camera"
)


camera_photo = None


if not st.session_state.camera_enabled:

    if st.sidebar.button(
        "📷 Open Camera",
        use_container_width=True
    ):

        st.session_state.camera_enabled = True
        st.rerun()

else:

    camera_photo = st.sidebar.camera_input(
        "📸 Take a Photo"
    )

    if st.sidebar.button(
        "❌ Close Camera",
        use_container_width=True
    ):

        st.session_state.camera_enabled = False
        st.rerun()


# ==========================================================
# COMBINE FILES
# ==========================================================

all_uploaded_files = (
    list(uploaded_files)
    if uploaded_files
    else []
)


if camera_photo is not None:

    all_uploaded_files.append(
        camera_photo
    )


st.sidebar.header(
    "🌐 Language & Speaker"
)


translation_language = st.sidebar.selectbox(
    "🌐 Image / Document Language:",
    EXPLANATION_LANGUAGES,
    key="common_translation_language"
)


listen_language = st.sidebar.selectbox(
    "🔊 Smart Reader Language:",
    list(SPEAKER_LANGUAGE_CODES.keys()),
    key="common_listen_language"
)




# ==========================================================
# FILE PROCESSING
# ==========================================================

if all_uploaded_files:

    pdf_files = [
        f
        for f in all_uploaded_files
        if f.name.lower().endswith(".pdf")
    ]

    doc_files = [
        f
        for f in all_uploaded_files
        if f.name.lower().endswith(".doc")
    ]

    docx_files = [
        f
        for f in all_uploaded_files
        if f.name.lower().endswith(".docx")
    ]

    image_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".heic",
        ".heif"
    )

    image_files = [
        f
        for f in all_uploaded_files
        if f.name.lower().endswith(
            image_extensions
        )
    ]

    document_files = (
        pdf_files +
        doc_files +
        docx_files
    )


    # ======================================================
    # DOCUMENT PROCESSING
    # ======================================================

    if document_files:

        current_document_names = [
            file.name
            for file in document_files
        ]

        if (
            current_document_names
            != st.session_state.processed_files
        ):

            with st.spinner(
                "⚙️ Auto-processing your study materials..."
            ):

                all_documents = []

                os.makedirs(
                    "data/uploaded_pdfs",
                    exist_ok=True
                )

                os.makedirs(
                    "data/converted_docs",
                    exist_ok=True
                )

                for uploaded_file in document_files:

                    original_name = uploaded_file.name

                    file_path = os.path.join(
                        "data/uploaded_pdfs",
                        original_name
                    )

                    with open(
                        file_path,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )


                    # PDF
                    if original_name.lower().endswith(
                        ".pdf"
                    ):

                        try:

                            loader = PyPDFLoader(
                                file_path
                            )

                            docs = loader.load()

                            all_documents.extend(
                                docs
                            )

                        except Exception as e:

                            st.sidebar.error(
                                f"❌ PDF error "
                                f"{original_name}: {e}"
                            )


                    # DOCX
                    elif original_name.lower().endswith(
                        ".docx"
                    ):

                        try:

                            loader = Docx2txtLoader(
                                file_path
                            )

                            docs = loader.load()

                            all_documents.extend(
                                docs
                            )

                        except Exception as e:

                            st.sidebar.error(
                                f"❌ DOCX error "
                                f"{original_name}: {e}"
                            )


                    # DOC
                    elif original_name.lower().endswith(
                        ".doc"
                    ):

                        try:

                            libreoffice_path = shutil.which(
                                "libreoffice"
                            )

                            if libreoffice_path is None:

                                libreoffice_path = shutil.which(
                                    "soffice"
                                )

                            if libreoffice_path is None:

                                raise Exception(
                                    "LibreOffice is not installed."
                                )

                            subprocess.run(
                                [
                                    libreoffice_path,
                                    "--headless",
                                    "--convert-to",
                                    "docx",
                                    "--outdir",
                                    "data/converted_docs",
                                    file_path
                                ],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                            )

                            converted_name = (
                                os.path.splitext(
                                    original_name
                                )[0]
                                + ".docx"
                            )

                            converted_path = os.path.join(
                                "data/converted_docs",
                                converted_name
                            )

                            if not os.path.exists(
                                converted_path
                            ):

                                raise Exception(
                                    "DOC to DOCX conversion failed."
                                )

                            loader = Docx2txtLoader(
                                converted_path
                            )

                            docs = loader.load()

                            all_documents.extend(
                                docs
                            )

                        except Exception as e:

                            st.sidebar.error(
                                f"❌ Could not process "
                                f"{original_name}: {e}"
                            )


                # CREATE VECTOR STORE
                if all_documents:

                    text_splitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                    )

                    splits = (
                        text_splitter.split_documents(
                            all_documents
                        )
                    )

                    embeddings = HuggingFaceEmbeddings(
                        model_name=(
                            "sentence-transformers/"
                            "all-MiniLM-L6-v2"
                        )
                    )

                    vector_store = (
                        FAISS.from_documents(
                            splits,
                            embeddings
                        )
                    )

                    st.session_state.vector_store = (
                        vector_store
                    )

                    os.makedirs(
                        "faiss_index",
                        exist_ok=True
                    )

                    vector_store.save_local(
                        "faiss_index"
                    )

                    st.session_state.processed_files = (
                        current_document_names
                    )

                    st.sidebar.success(
                        f"✅ Automatically processed "
                        f"{len(document_files)} document(s)!"
                    )

                else:

                    st.sidebar.error(
                        "❌ No readable text was found."
                    )

        else:

            st.sidebar.success(
                f"✅ {len(document_files)} "
                f"document(s) ready for chat!"
            )


    # ======================================================
    # IMAGE PROCESSING
    # ======================================================

    if image_files:

        st.sidebar.markdown("---")

        st.sidebar.subheader(
            "📸 Uploaded Image"
        )

        img_to_process = image_files[0]

        st.sidebar.image(
            img_to_process,
            caption=img_to_process.name,
            use_container_width=True
        )


        # ==================================================
        # IMAGE SENTENCE COUNT
        # ==================================================

        image_sentence_count = st.sidebar.selectbox(
            "📝 Image Explanation Length:",
            list(range(1, 11)),
            index=2,
            key="image_sentence_count"
        )


        # ==================================================
        # SHOW SELECTED LANGUAGE
        # ==================================================

        st.sidebar.info(
            f"🌐 Image language: {translation_language}"
        )


        # ==================================================
        # ANALYZE IMAGE
        # ==================================================

        if st.sidebar.button(
            "🔍 Analyze Image",
            use_container_width=True,
            key="analyze_image_button"
        ):

            with st.spinner(
                "AI is looking at your image..."
            ):

                try:

                    image_bytes = (
                        img_to_process.getvalue()
                    )

                    if not image_bytes:

                        raise ValueError(
                            "Image data is empty."
                        )

                    image_base64 = (
                        base64.b64encode(
                            image_bytes
                        ).decode("utf-8")
                    )

                    mime_type = (
                        img_to_process.type
                        or "image/jpeg"
                    )


                    # ==========================================
                    # LANGUAGE-SPECIFIC IMAGE INSTRUCTION
                    # ==========================================

                    image_language_instruction = (
                        get_language_instruction(
                            translation_language
                        )
                    )


                    # ==========================================
                    # IMAGE PROMPT
                    # ==========================================

                    image_prompt = f"""
Analyze the provided image.

Generate EXACTLY {image_sentence_count} sentences
describing the visible content.

TARGET LANGUAGE:
{translation_language}

LANGUAGE RULES:

{image_language_instruction}

STRICT IMAGE RULES:

1. Exactly {image_sentence_count} sentences.
2. Not more than {image_sentence_count}.
3. Not fewer than {image_sentence_count}.
4. Describe ONLY what is actually visible in the image.
5. Do not guess hidden information.
6. Use simple student-friendly language.
7. No heading.
8. No greeting.
9. No introduction.
10. No markdown.
11. No bullets.
12. No numbering.
13. No emojis.
14. Do not write analysis.
15. Do not write reasoning.
16. Do not write <think>.
17. Do not mention these instructions.
18. Do not mention the model.
19. Return ONLY the final explanation.
20. Every sentence must be in {translation_language}.

IMPORTANT FOR HINDI:

If TARGET LANGUAGE is Hindi:
- Write Hindi using Devanagari script.
- Do NOT write Hindi in Roman/English letters.
- Do NOT use Hinglish.
- Example of correct Hindi: "इस तस्वीर में एक कमरा दिखाई दे रहा है।"
- Example of WRONG Hindi: "Is tasveer mein ek kamra dikhai de raha hai."
- The final answer must be predominantly Devanagari Hindi.

IMPORTANT FOR HINGLISH:

If TARGET LANGUAGE is Hinglish:
- Use natural Roman Hindi mixed with English.
- Do not force pure Hindi.

Return exactly {image_sentence_count} sentences.

TARGET LANGUAGE:
{translation_language}
"""


                    # ==========================================
                    # VISION MODEL
                    # ==========================================

                    vision_llm = ChatGroq(
                        api_key=st.secrets[
                            "GROQ_API_KEY"
                        ],
                        model_name="qwen/qwen3.6-27b",
                        temperature=0
                    )


                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": image_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":
                                    f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    )


                    response = vision_llm.invoke(
                        [message]
                    )


                    # ==========================================
                    # SAFE RESPONSE EXTRACTION
                    # ==========================================

                    image_explanation = getattr(
                        response,
                        "content",
                        ""
                    )


                    if isinstance(
                        image_explanation,
                        list
                    ):

                        parts = []

                        for item in image_explanation:

                            if isinstance(
                                item,
                                dict
                            ):

                                if "text" in item:

                                    parts.append(
                                        str(
                                            item["text"]
                                        )
                                    )

                            else:

                                parts.append(
                                    str(item)
                                )

                        image_explanation = " ".join(
                            parts
                        )


                    image_explanation = str(
                        image_explanation or ""
                    ).strip()


                    # ==========================================
                    # REMOVE THINKING
                    # ==========================================

                    image_explanation = remove_thinking(
                        image_explanation
                    )


                    # ==========================================
                    # REMOVE CODE FENCES
                    # ==========================================

                    image_explanation = re.sub(
                        r"```.*?```",
                        "",
                        image_explanation,
                        flags=re.DOTALL
                    ).strip()


                    # ==========================================
                    # REMOVE PREFIX
                    # ==========================================

                    image_explanation = re.sub(
                        r"^(answer|response|description)\s*:\s*",
                        "",
                        image_explanation,
                        flags=re.IGNORECASE
                    ).strip()


                    if not image_explanation:

                        raise ValueError(
                            "Image model returned an empty response."
                        )


                    # ==========================================
                    # SENTENCE SPLITTING
                    #
                    # Supports:
                    # English .
                    # Hindi ।
                    # Bengali etc.
                    # ==========================================

                    image_sentences = re.split(
                        r'(?<=[.!?।॥])\s+',
                        image_explanation
                    )


                    image_sentences = [
                        sentence.strip()
                        for sentence in image_sentences
                        if sentence.strip()
                    ]


                    # ==========================================
                    # LINE FALLBACK
                    # ==========================================

                    if (
                        len(image_sentences)
                        < image_sentence_count
                    ):

                        line_sentences = []

                        for line in image_explanation.splitlines():

                            line = line.strip()

                            if line:

                                line_sentences.append(
                                    line
                                )

                        if (
                            len(line_sentences)
                            >= image_sentence_count
                        ):

                            image_sentences = (
                                line_sentences
                            )


                    # ==========================================
                    # TOO MANY SENTENCES
                    # ==========================================

                    if (
                        len(image_sentences)
                        > image_sentence_count
                    ):

                        image_sentences = (
                            image_sentences[
                                :image_sentence_count
                            ]
                        )


                    # ==========================================
                    # FINAL SENTENCE VALIDATION
                    # ==========================================

                    if (
                        len(image_sentences)
                        != image_sentence_count
                    ):

                        raise ValueError(
                            f"AI generated "
                            f"{len(image_sentences)} "
                            f"sentences instead of "
                            f"{image_sentence_count}."
                        )


                    image_explanation = " ".join(
                        image_sentences
                    )


                    # ==========================================
                    # EXTRA HINDI SAFETY
                    #
                    # If Hindi selected but model still
                    # returns Roman Hindi, translate it again.
                    # ==========================================

                    if translation_language == "Hindi":

                        devanagari_count = len(
                            re.findall(
                                r"[\u0900-\u097F]",
                                image_explanation
                            )
                        )

                        latin_count = len(
                            re.findall(
                                r"[A-Za-z]",
                                image_explanation
                            )
                        )

                        # If Roman text dominates,
                        # force a second Hindi conversion.
                        if (
                            latin_count > devanagari_count
                        ):

                            hindi_prompt = f"""
Convert the following explanation into PURE HINDI.

VERY IMPORTANT:
- Use Devanagari script only.
- Do not use Roman Hindi.
- Do not use Hinglish.
- Do not add or remove information.
- Keep EXACTLY {image_sentence_count} sentences.
- Describe only the same image information.
- Return ONLY the Hindi sentences.

TEXT:

{image_explanation}
"""

                            hindi_llm = ChatGroq(
                                api_key=st.secrets[
                                    "GROQ_API_KEY"
                                ],
                                model_name="qwen/qwen3.6-27b",
                                temperature=0
                            )

                            hindi_response = (
                                hindi_llm.invoke(
                                    [
                                        HumanMessage(
                                            content=hindi_prompt
                                        )
                                    ]
                                )
                            )

                            image_explanation = remove_thinking(
                                getattr(
                                    hindi_response,
                                    "content",
                                    ""
                                )
                            ).strip()

                            image_explanation = re.sub(
                                r"```.*?```",
                                "",
                                image_explanation,
                                flags=re.DOTALL
                            ).strip()


                    # ==========================================
                    # SAVE IMAGE EXPLANATION
                    # ==========================================

                    st.session_state.image_explanation = (
                        image_explanation
                    )


                    # ==========================================
                    # CLEAR TRANSLATION CACHE
                    # ==========================================

                    st.session_state.image_speaker_text = None

                    st.session_state.image_speaker_text_language = None

                    st.session_state.speaker_text = None

                    st.session_state.speaker_text_language = None


                    # ==========================================
                    # IMAGE NAME + LANGUAGE TRACKING
                    # ==========================================

                    current_image_key = (
                        f"{img_to_process.name}"
                        f"__{translation_language}"
                    )


                    # ==========================================
                    # SAVE TO CHAT ONLY ONCE
                    # ==========================================

                    if (
                        st.session_state.analyzed_image_name
                        != current_image_key
                    ):

                        st.session_state.messages.append({
                            "role": "user",
                            "content":
                            f"📸 User uploaded image: "
                            f"{img_to_process.name}"
                        })


                        st.session_state.messages.append({
                            "role": "assistant",
                            "content":
                            image_explanation
                        })


                        st.session_state.analyzed_image_name = (
                            current_image_key
                        )


                    st.success(
                        f"✅ Image explanation generated in "
                        f"{translation_language}!"
                    )


                    st.rerun()


                except Exception as e:

                    st.sidebar.error(
                        f"❌ Error analyzing image: {e}"
                    )


# ==========================================================
# DOCUMENT STUDY TOOLS
# ==========================================================

if st.session_state.vector_store is not None:

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "🎓 Document Study Tools"
    )


    explanation_level = st.sidebar.selectbox(
        "📊 Difficulty Level:",
        [
            "Easy",
            "Medium",
            "Hard"
        ],
        key="document_explanation_level"
    )


    exam_points = st.sidebar.checkbox(
        "🎯 Include Exam Important Points",
        value=True,
        key="document_exam_points"
    )


    language_instruction = get_language_instruction(
        translation_language
    )


    if explanation_level == "Easy":

        level_instruction = """
Explain everything for a beginner.

Use very simple language,
short explanations and easy examples.
"""

    elif explanation_level == "Medium":

        level_instruction = """
Explain at normal school/college level.

Cover important concepts,
definitions and useful examples.
"""

    else:

        level_instruction = """
Explain the topic deeply.

Include technical details,
advanced concepts,
important relationships and examples.
"""


    if exam_points:

        exam_instruction = """
At the end include:

Exam Important Points

Include:
- Important definitions
- Important concepts
- Important formulas
- Important facts
- Possible exam questions
"""

    else:

        exam_instruction = ""


    # ======================================================
    # EXPLAIN DOCUMENT
    # ======================================================

    if st.sidebar.button(
        "✨ Explain Document",
        use_container_width=True,
        key="explain_document_button"
    ):

        with st.spinner(
            "🤖 AI is preparing your explanation..."
        ):

            try:

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        "Explain all important topics concepts "
                        "definitions examples and exam "
                        "important points.",
                        k=12
                    )
                )


                document_context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )


                explanation_prompt = f"""
You are an expert teacher and AI Study Buddy.

Explain the uploaded study material.

TARGET LANGUAGE:
{translation_language}

{language_instruction}

{level_instruction}

{exam_instruction}

IMPORTANT RULES:

1. Only use information from the study material.
2. Do not invent facts.
3. Do not mention that you are an AI.
4. Do not add unnecessary greetings.
5. Do not add irrelevant information.
6. Explain clearly for a student.
7. Use proper paragraphs.
8. Keep the explanation useful for studying.
9. Return the answer in the selected language.

Study Material:

{document_context}
"""


                study_llm = ChatGroq(
                    api_key=st.secrets[
                        "GROQ_API_KEY"
                    ],
                    model_name="qwen/qwen3.6-27b",
                    temperature=0
                )


                response = study_llm.invoke(
                    [
                        HumanMessage(
                            content=explanation_prompt
                        )
                    ]
                )


                explanation = getattr(
                    response,
                    "content",
                    ""
                )


                explanation = remove_thinking(
                    explanation
                )


                st.session_state.document_explanation = (
                    explanation
                )


                st.session_state.document_speaker_text = None

                st.session_state.document_speaker_text_language = None

                st.session_state.speaker_text = None

                st.session_state.speaker_text_language = None


                st.success(
                    f"✅ Document explanation generated "
                    f"in {translation_language}!"
                )


            except Exception as e:

                st.error(
                    f"❌ Explanation error: {e}"
                )


# ==========================================================
# CURRENT READER SOURCE
# ==========================================================

reader_source_text = None

reader_source_type = None


if st.session_state.get(
    "document_explanation"
):

    reader_source_text = (
        st.session_state.document_explanation
    )

    reader_source_type = "Document"

elif st.session_state.get(
    "image_explanation"
):

    reader_source_text = (
        st.session_state.image_explanation
    )

    reader_source_type = "Image"


# ==========================================================
# DISPLAY DOCUMENT EXPLANATION
# ==========================================================

if st.session_state.get(
    "document_explanation"
):

    st.markdown("---")

    st.subheader(
        "📚 AI Document Explanation"
    )

    st.caption(
        f"Language: {translation_language}"
    )

    st.markdown(
        remove_thinking(
            st.session_state.document_explanation
        )
    )


# ==========================================================
# DISPLAY IMAGE EXPLANATION
# ==========================================================

if st.session_state.get(
    "image_explanation"
):

    st.markdown("---")

    st.subheader(
        "🖼️ AI Image Explanation"
    )

    st.caption(
        f"Language: {translation_language}"
    )

    st.markdown(
        remove_thinking(
            st.session_state.image_explanation
        )
    )


# ==========================================================
# COMMON SMART READER
# ==========================================================

if reader_source_text:

    st.markdown("---")

    st.subheader(
        f"🔊📖 Smart Reader — {reader_source_type}"
    )

    st.caption(
        f"Reading language: {listen_language}"
    )

    st.caption(
        "Choose the reading language and click any "
        "sentence to start reading from there."
    )


    # ======================================================
    # SELECT CACHE
    # ======================================================

    if reader_source_type == "Image":

        cached_text = (
            st.session_state.image_speaker_text
        )

        cached_language = (
            st.session_state.image_speaker_text_language
        )

    else:

        cached_text = (
            st.session_state.document_speaker_text
        )

        cached_language = (
            st.session_state.document_speaker_text_language
        )


    # ======================================================
    # PREPARE SPEECH TEXT
    # ======================================================

    if listen_language == translation_language:

        speech_text = clean_text_for_speech(
            reader_source_text
        )


    elif (
        cached_text
        and
        cached_language == listen_language
    ):

        speech_text = cached_text


    else:

        speech_text = ""

        if st.button(
            f"🌐 Prepare {listen_language} Reading",
            key=(
                f"prepare_reader_translation_"
                f"{reader_source_type}"
            )
        ):

            with st.spinner(
                f"🌐 Translating into "
                f"{listen_language}..."
            ):

                try:

                    translated_text = (
                        translate_for_speech(
                            reader_source_text,
                            listen_language
                        )
                    )


                    if not translated_text:

                        raise ValueError(
                            "Translation returned empty text."
                        )


                    # ==========================================
                    # SAVE CORRECT CACHE
                    # ==========================================

                    if reader_source_type == "Image":

                        st.session_state.image_speaker_text = (
                            translated_text
                        )

                        st.session_state.image_speaker_text_language = (
                            listen_language
                        )

                    else:

                        st.session_state.document_speaker_text = (
                            translated_text
                        )

                        st.session_state.document_speaker_text_language = (
                            listen_language
                        )


                    st.session_state.speaker_text = (
                        translated_text
                    )

                    st.session_state.speaker_text_language = (
                        listen_language
                    )


                    st.success(
                        f"✅ {listen_language} "
                        f"reading ready!"
                    )


                    st.rerun()


                except Exception as e:

                    st.error(
                        f"❌ Translation error: {e}"
                    )


    # ======================================================
    # SMART READER
    # ======================================================

    if speech_text:

        # IMPORTANT:
        # Hindi uses both "." and "।"
        # This splitter handles both.

        sentences = re.split(
            r'(?<=[.!?।॥])\s+',
            speech_text
        )


        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]


        sentences_json = json.dumps(
            sentences,
            ensure_ascii=False
        )


        language_code = (
            SPEAKER_LANGUAGE_CODES[
                listen_language
            ]
        )


        reader_html = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

.reader-box {
    background: #f8fafc;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 18px;
    font-family: Arial, "Noto Sans Devanagari", sans-serif;
    line-height: 1.9;
    color: #111827;
    max-height: 450px;
    overflow-y: auto;
}

.reader-sentence {
    padding: 3px 5px;
    border-radius: 5px;
    cursor: pointer;
    display: inline;
}

.reader-sentence:hover {
    background: #dbeafe;
}

.reader-sentence.active {
    background: #fde68a;
    border-bottom: 3px solid #f59e0b;
}

.reader-controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.reader-button {
    background: #2563eb;
    color: white;
    border: none;
    padding: 9px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
}

.reader-button:hover {
    background: #1d4ed8;
}

.reader-status {
    margin-top: 10px;
    color: #475569;
    font-size: 13px;
}

</style>

</head>

<body>

<div class="reader-controls">

<button class="reader-button"
        onclick="startReader()">
▶️ Start
</button>

<button class="reader-button"
        onclick="pauseReader()">
⏸️ Pause
</button>

<button class="reader-button"
        onclick="resumeReader()">
▶️ Resume
</button>

<button class="reader-button"
        onclick="stopReader()">
⏹️ Stop
</button>

</div>

<div id="readerBox"
     class="reader-box">
</div>

<div id="readerStatus"
     class="reader-status">
Ready to read
</div>


<script>

const readerSentences =
    __SENTENCES__;

const readerLanguage =
    "__LANGUAGE__";


const readerBox =
    document.getElementById(
        "readerBox"
    );


const readerStatus =
    document.getElementById(
        "readerStatus"
    );


let currentSentence = 0;


// ==========================================================
// FIND BEST VOICE
// ==========================================================

function getBestVoice(language) {

    const voices =
        window.speechSynthesis.getVoices();

    if (!voices || voices.length === 0) {
        return null;
    }

    const target =
        language.toLowerCase();

    // Exact language match first
    let voice =
        voices.find(
            function(v) {

                return (
                    v.lang &&
                    v.lang.toLowerCase() === target
                );

            }
        );

    if (voice) {
        return voice;
    }


    // Hindi fallback
    if (target.startsWith("hi")) {

        voice =
            voices.find(
                function(v) {

                    return (
                        v.lang &&
                        v.lang.toLowerCase().startsWith("hi")
                    );

                }
            );

        if (voice) {
            return voice;
        }
    }


    // General language fallback
    const shortLanguage =
        target.split("-")[0];

    voice =
        voices.find(
            function(v) {

                return (
                    v.lang &&
                    v.lang.toLowerCase().startsWith(
                        shortLanguage
                    )
                );

            }
        );

    return voice || null;
}


// ==========================================================
// LOAD VOICES
// ==========================================================

window.speechSynthesis.onvoiceschanged =
    function() {

        window.speechSynthesis.getVoices();

    };


// ==========================================================
// CREATE SENTENCES
// ==========================================================

readerSentences.forEach(
    function(sentence, index) {

        const span =
            document.createElement(
                "span"
            );

        span.className =
            "reader-sentence";

        span.textContent =
            sentence;

        span.onclick =
            function() {

                startFrom(index);

            };

        readerBox.appendChild(
            span
        );

        readerBox.appendChild(
            document.createTextNode(" ")
        );

    }
);


// ==========================================================
// CLEAR HIGHLIGHT
// ==========================================================

function clearHighlight() {

    document
        .querySelectorAll(
            ".reader-sentence"
        )
        .forEach(
            function(element) {

                element.classList.remove(
                    "active"
                );

            }
        );

}


// ==========================================================
// START FROM CLICKED SENTENCE
// ==========================================================

function startFrom(index) {

    window.speechSynthesis.cancel();

    currentSentence = index;

    speakCurrent();

}


// ==========================================================
// START
// ==========================================================

function startReader() {

    window.speechSynthesis.cancel();

    currentSentence = 0;

    speakCurrent();

}


// ==========================================================
// SPEAK CURRENT
// ==========================================================

function speakCurrent() {

    if (
        currentSentence >=
        readerSentences.length
    ) {

        clearHighlight();

        readerStatus.textContent =
            "✅ Reading completed";

        return;

    }


    clearHighlight();


    const elements =
        document.querySelectorAll(
            ".reader-sentence"
        );


    const current =
        elements[currentSentence];


    if (current) {

        current.classList.add(
            "active"
        );

        current.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }


    const utterance =
        new SpeechSynthesisUtterance(
            readerSentences[
                currentSentence
            ]
        );


    // IMPORTANT:
    // Selected language is passed directly
    // to browser speech engine.

    utterance.lang =
        readerLanguage;


    // Find Hindi/English/etc voice
    const selectedVoice =
        getBestVoice(
            readerLanguage
        );


    if (selectedVoice) {

        utterance.voice =
            selectedVoice;

    }


    utterance.rate =
        0.90;


    utterance.pitch =
        1.0;


    readerStatus.textContent =
        "🔊 Reading sentence "
        + (currentSentence + 1)
        + " of "
        + readerSentences.length;


    utterance.onend =
        function() {

            currentSentence++;

            speakCurrent();

        };


    utterance.onerror =
        function(event) {

            readerStatus.textContent =
                "⚠️ Speech could not be played. "
                + "Please check that your browser "
                + "supports "
                + readerLanguage
                + " voice.";

        };


    window.speechSynthesis.speak(
        utterance
    );

}


// ==========================================================
// PAUSE
// ==========================================================

function pauseReader() {

    window.speechSynthesis.pause();

    readerStatus.textContent =
        "⏸️ Reading paused";

}


// ==========================================================
// RESUME
// ==========================================================

function resumeReader() {

    window.speechSynthesis.resume();

    readerStatus.textContent =
        "▶️ Reading resumed";

}


// ==========================================================
// STOP
// ==========================================================

function stopReader() {

    window.speechSynthesis.cancel();

    clearHighlight();

    currentSentence = 0;

    readerStatus.textContent =
        "⏹️ Reading stopped";

}

</script>

</body>

</html>
"""


        reader_html = reader_html.replace(
            "__SENTENCES__",
            sentences_json
        )


        reader_html = reader_html.replace(
            "__LANGUAGE__",
            language_code
        )


        components.html(
            reader_html,
            height=600,
            scrolling=True
        )


# ==========================================================
# SMART STUDY TOOLS
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🧠 Smart Study Tools"
)


# ==========================================================
# QUIZ
# ==========================================================

num_questions = st.sidebar.slider(
    "How many questions?",
    min_value=1,
    max_value=100,
    value=5,
    key="quiz_question_count"
)


if st.sidebar.button(
    "📝 Generate MCQ Quiz",
    key="generate_quiz_button"
):

    if st.session_state.vector_store is not None:

        with st.spinner(
            f"AI is preparing your "
            f"{num_questions}-question quiz..."
        ):

            try:

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        "core concepts definitions "
                        "important topics",
                        k=10
                    )
                )


                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )


                quiz_prompt = f"""
You are an expert teacher.

Based ONLY on the following study material,
create exactly {num_questions} multiple-choice questions.

STUDY MATERIAL:

{context}

Return ONLY valid JSON.

Required format:

[
  {{
    "question": "Question here",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Correct option"
  }}
]

Rules:

- Exactly {num_questions} questions.
- Exactly 4 options per question.
- The answer must exactly match one option.
- No markdown.
- No explanation.
- No text before or after JSON.
"""


                raw_text = ask_llm(
                    quiz_prompt
                )


                clean_text = (
                    clean_json_response(
                        raw_text
                    )
                )


                quiz_data = json.loads(
                    clean_text
                )


                if not isinstance(
                    quiz_data,
                    list
                ):

                    raise ValueError(
                        "Quiz response is not a list."
                    )


                valid_quiz = []


                for question in quiz_data:

                    if not isinstance(
                        question,
                        dict
                    ):
                        continue

                    if (
                        "question" not in question
                        or
                        "options" not in question
                        or
                        "answer" not in question
                    ):
                        continue

                    if len(
                        question["options"]
                    ) != 4:
                        continue

                    if (
                        question["answer"]
                        not in question["options"]
                    ):
                        continue

                    valid_quiz.append(
                        question
                    )


                if len(valid_quiz) < num_questions:

                    raise ValueError(
                        "AI did not generate enough valid questions."
                    )


                st.session_state.quiz_data = (
                    valid_quiz[:num_questions]
                )


                st.sidebar.success(
                    "✅ Quiz generated!"
                )


                st.rerun()


            except Exception as e:

                st.sidebar.error(
                    f"Quiz generation failed: {e}"
                )

    else:

        st.sidebar.error(
            "Please upload and process a document first!"
        )


# ==========================================================
# SUMMARY
# ==========================================================

if st.sidebar.button(
    "📄 Generate Summary",
    key="generate_summary_button"
):

    if st.session_state.vector_store is not None:

        with st.spinner(
            "AI is reading and summarizing..."
        ):

            try:

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        "comprehensive summary "
                        "core themes definitions",
                        k=8
                    )
                )


                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )


                summary_prompt = f"""
You are an expert teacher.

Create a highly structured study summary
from the following material.

TARGET LANGUAGE:
{translation_language}

{get_language_instruction(translation_language)}

MATERIAL:

{context}

Requirements:

- Clear headings
- Bullet points
- Important definitions
- Important concepts
- Exam-focused points
- Easy language
- No unnecessary introduction
"""


                summary_result = ask_llm(
                    summary_prompt
                )


                summary_result = remove_thinking(
                    summary_result
                )


                st.session_state.messages.append({
                    "role": "assistant",
                    "content":
                    "## 📚 Quick Revision Summary\n\n"
                    + summary_result
                })


                st.rerun()


            except Exception as e:

                st.sidebar.error(
                    f"Summary generation failed: {e}"
                )

    else:

        st.sidebar.error(
            "Please upload and process a document first!"
        )


# ==========================================================
# FIND STUDY TOPICS
# ==========================================================

if st.sidebar.button(
    "📚 Find Study Topics",
    key="find_topics_button"
):

    if st.session_state.vector_store is not None:

        with st.spinner(
            "Finding important topics..."
        ):

            try:

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        "main topics chapters concepts "
                        "headings important subjects",
                        k=12
                    )
                )


                context = "\n\n".join(
                    str(doc.page_content)
                    for doc in docs
                )


                topic_prompt = f"""
You are an educational content analyzer.

Analyze the study material below and identify
the most important topics that a student should study.

STUDY MATERIAL:

{context}

RULES:

1. Return ONLY a numbered list.
2. Give 5 to 10 important topics.
3. Each topic must be short.
4. Do not give explanations.
5. Do not use markdown headings.
6. Do not repeat topics.
7. Use topics that actually appear in or are strongly
   related to the study material.
"""


                raw_response = ask_llm(
                    topic_prompt
                )


                raw_text = remove_thinking(
                    str(raw_response)
                ).strip()


                topics = []


                for line in raw_text.splitlines():

                    line = line.strip()

                    if not line:
                        continue


                    line = re.sub(
                        r"^\s*\d+[\.\)\-:]\s*",
                        "",
                        line
                    )


                    line = re.sub(
                        r"^\s*[-*•]\s*",
                        "",
                        line
                    )


                    line = line.strip()


                    if (
                        line
                        and
                        len(line) > 2
                        and
                        len(line) <= 100
                    ):

                        if line not in topics:

                            topics.append(
                                line
                            )


                topics = topics[:10]


                if not topics:

                    raise ValueError(
                        "Could not identify study topics."
                    )


                st.session_state.mindmap_topics = topics

                st.session_state.selected_mindmap_topic = (
                    topics[0]
                )


                st.sidebar.success(
                    f"Found {len(topics)} study topics!"
                )


            except Exception as e:

                st.sidebar.error(
                    f"Topic detection failed: {e}"
                )

    else:

        st.sidebar.error(
            "Please upload and process a document first!"
        )


# ==========================================================
# SELECT TOPIC
# ==========================================================

if st.session_state.mindmap_topics:

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "🧠 Select a Topic"
    )


    selected_topic = st.sidebar.selectbox(
        "Choose a topic for your Mind Map:",
        st.session_state.mindmap_topics,
        key="selected_mindmap_topic"
    )


    # ======================================================
    # GENERATE MIND MAP
    # ======================================================

    if st.sidebar.button(
        "🗺️ Generate Mind Map",
        key="generate_mindmap_button"
    ):

        if st.session_state.vector_store is not None:

            with st.spinner(
                f"Designing Mind Map for "
                f"{selected_topic}..."
            ):

                try:

                    docs = (
                        st.session_state.vector_store
                        .similarity_search(
                            selected_topic,
                            k=10
                        )
                    )


                    context = "\n\n".join(
                        str(doc.page_content)
                        for doc in docs
                    )


                    mindmap_prompt = f"""
You are an expert educational mind-map designer.

Create a VALID Graphviz DOT mind map ONLY for:

TOPIC:
{selected_topic}

STUDY MATERIAL:
{context}

STRICT RULES:

1. Return ONLY valid Graphviz DOT code.
2. Do NOT use markdown.
3. Do NOT use code fences.
4. Do NOT write any explanation.
5. The first word MUST be digraph.
6. Use exactly:
   digraph G {{
7. Use rankdir=LR.
8. Create one central topic.
9. Connect the central topic to major concepts.
10. Add useful subtopics.
11. Maximum 15 nodes.
12. Every node label must be inside double quotes.
13. Every edge must use:
    "Parent" -> "Child";
14. Do not put double quotes inside node labels.
15. Avoid special characters that can break DOT syntax.
16. Keep labels short and readable.
"""


                    raw_response = ask_llm(
                        mindmap_prompt
                    )


                    raw_text = remove_thinking(
                        str(raw_response)
                    ).strip()


                    raw_text = re.sub(
                        r"```(?:dot|graphviz)?",
                        "",
                        raw_text,
                        flags=re.IGNORECASE
                    )


                    raw_text = raw_text.replace(
                        "```",
                        ""
                    ).strip()


                    match = re.search(
                        r"\bdigraph\b",
                        raw_text,
                        flags=re.IGNORECASE
                    )


                    if not match:

                        raise ValueError(
                            "AI did not return Graphviz DOT code."
                        )


                    clean_dot = raw_text[
                        match.start():
                    ].strip()


                    end_idx = clean_dot.rfind(
                        "}"
                    )


                    if end_idx == -1:

                        raise ValueError(
                            "Graphviz code is incomplete."
                        )


                    clean_dot = clean_dot[
                        :end_idx + 1
                    ].strip()


                    if not clean_dot.lower().startswith(
                        "digraph"
                    ):

                        raise ValueError(
                            "Invalid Graphviz format."
                        )


                    st.markdown("---")

                    st.subheader(
                        f"🗺️ Mind Map: {selected_topic}"
                    )


                    st.graphviz_chart(
                        clean_dot,
                        use_container_width=True
                    )


                except Exception as e:

                    st.sidebar.error(
                        f"Mind Map failed: {e}"
                    )

        else:

            st.sidebar.error(
                "Please upload and process a document first!"
            )


# ==========================================================
# FLASHCARDS
# ==========================================================

if st.sidebar.button(
    "🗂️ Generate Flashcards",
    key="generate_flashcards_button"
):

    if st.session_state.vector_store is None:

        st.sidebar.error(
            "Please upload a document first!"
        )

    else:

        with st.spinner(
            "Creating Flashcards..."
        ):

            try:

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        "important terms definitions "
                        "key concepts",
                        k=5
                    )
                )


                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )


                prompt = f"""
Create exactly 6 important flashcards
from this study material:

{context}

Return ONLY valid JSON in this format:

[
  {{
    "term": "Term",
    "definition": "Simple definition"
  }}
]

No markdown.
No explanation.
"""


                response = llm.invoke([
                    HumanMessage(
                        content=prompt
                    )
                ])


                text = remove_thinking(
                    response.content
                ).strip()

                text = text.replace(
                    "```json",
                    ""
                )

                text = text.replace(
                    "```",
                    ""
                ).strip()


                start = text.find("[")

                end = text.rfind("]") + 1


                cards = json.loads(
                    text[start:end]
                )


                st.session_state.flashcards = (
                    cards[:6]
                )


                st.sidebar.success(
                    "✅ Flashcards ready!"
                )


                st.rerun()


            except Exception as e:

                st.sidebar.error(
                    f"Flashcard error: {e}"
                )


# ==========================================================
# DISPLAY FLASHCARDS
# ==========================================================

if st.session_state.flashcards:

    st.markdown("---")

    st.subheader(
        "🗂️ Interactive Flashcards"
    )

    st.caption(
        "Hover over a card to flip it 👆"
    )


    cols = st.columns(3)


    for i, card in enumerate(
        st.session_state.flashcards
    ):

        term = str(
            card.get(
                "term",
                ""
            )
        )


        definition = str(
            card.get(
                "definition",
                ""
            )
        )


        term_html = (
            term
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


        definition_html = (
            definition
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


        card_html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

body {{
    margin: 0;
    padding: 0;
    background: transparent;
}}

.flip-card {{
    width: 100%;
    height: 170px;
    perspective: 1000px;
}}

.flip-card-inner {{
    position: relative;
    width: 100%;
    height: 100%;
    transition: transform 0.6s;
    transform-style: preserve-3d;
}}

.flip-card:hover .flip-card-inner {{
    transform: rotateY(180deg);
}}

.flip-card-front,
.flip-card-back {{
    position: absolute;
    width: 100%;
    height: 100%;
    box-sizing: border-box;

    display: flex;
    align-items: center;
    justify-content: center;

    padding: 20px;
    text-align: center;

    border-radius: 12px;
    backface-visibility: hidden;

    font-family: Arial, sans-serif;
}}

.flip-card-front {{
    background-color: #2e2e2e;
    color: white;
    font-size: 22px;
    font-weight: bold;
}}

.flip-card-back {{
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    transform: rotateY(180deg);
}}

</style>

</head>

<body>

<div class="flip-card">

<div class="flip-card-inner">

<div class="flip-card-front">
{term_html}
</div>

<div class="flip-card-back">
{definition_html}
</div>

</div>

</div>

</body>

</html>
"""


        with cols[i % 3]:

            components.html(
                card_html,
                height=190,
                scrolling=False
            )


# ==========================================================
# QUIZ DISPLAY
# ==========================================================

if st.session_state.quiz_data:

    st.markdown("---")

    st.subheader(
        "📝 Your Interactive Quiz"
    )


    with st.form(
        "interactive_quiz_form"
    ):

        user_answers = []


        for i, q in enumerate(
            st.session_state.quiz_data
        ):

            st.markdown(
                f"**Q{i + 1}: "
                f"{q['question']}**"
            )


            ans = st.radio(
                f"Select option for Q{i + 1}:",
                q["options"],
                key=f"radio_q{i}",
                label_visibility="collapsed",
                index=None
            )


            user_answers.append(
                ans
            )


            st.write("")


        submitted = st.form_submit_button(
            "Submit Answers"
        )


    if submitted:

        score = 0
        wrong = 0


        st.markdown(
            "### 📊 Quiz Results"
        )


        for i, q in enumerate(
            st.session_state.quiz_data
        ):

            if user_answers[i] == q["answer"]:

                score += 1

                st.success(
                    f"**Q{i + 1}: Correct!** "
                    f"({q['answer']})"
                )


            elif user_answers[i] is None:

                wrong += 1

                st.warning(
                    f"**Q{i + 1}: Not Answered.** "
                    f"Correct answer: "
                    f"'{q['answer']}'"
                )


            else:

                wrong += 1

                st.error(
                    f"**Q{i + 1}: Incorrect.** "
                    f"You chose "
                    f"'{user_answers[i]}'. "
                    f"Correct answer: "
                    f"'{q['answer']}'"
                )


        total_questions = len(
            st.session_state.quiz_data
        )


        st.info(
            f"**Final Score: {score} Correct, "
            f"{wrong} Incorrect out of "
            f"{total_questions}.**"
        )


        if st.button(
            "❌ Close Quiz",
            key="close_quiz_button"
        ):

            st.session_state.quiz_data = None

            st.rerun()


# ==========================================================
# CHAT SETTINGS
# ==========================================================

st.markdown("---")

st.sidebar.title(
    "⚙️ Chat Settings"
)


font_size = st.sidebar.slider(
    "Font size",
    min_value=12,
    max_value=30,
    value=16,
    key="chat_font_size"
)


st.markdown(
    f"""
<style>

.stChatMessage p,
.stChatMessage li,
.stChatMessage span {{
    font-size: {font_size}px !important;
}}

</style>
""",
    unsafe_allow_html=True
)


# ==========================================================
# CHAT VOICE LANGUAGE
# ==========================================================

lang_option = st.sidebar.selectbox(
    "🗣️ Choose AI Voice Language:",
    [
        "English",
        "Hindi"
    ],
    key="chat_voice_language"
)


selected_lang = (
    "hi"
    if lang_option == "Hindi"
    else "en"
)


# ==========================================================
# CLEAR CHAT
# ==========================================================

if st.sidebar.button(
    "🗑️ Clear Chat History",
    key="clear_chat_button"
):

    st.session_state.messages = []

    st.rerun()


# ==========================================================
# MANUAL TEXT TO SPEECH
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.title(
    "Text to Speech App"
)


sidebar_text = st.sidebar.text_area(
    "Write your text here:",
    "",
    key="manual_tts_text"
)


if st.sidebar.button(
    "Play Audio",
    key="manual_tts_button"
):

    if sidebar_text.strip():

        try:

            clean_sidebar_text = (
                clean_text_for_speech(
                    sidebar_text
                )
            )


            tts = gTTS(
                text=clean_sidebar_text,
                lang=selected_lang
            )


            tts.save(
                "response.mp3"
            )


            st.sidebar.audio(
                "response.mp3",
                format="audio/mp3"
            )


        except Exception as e:

            st.sidebar.error(
                f"Error: {e}"
            )

    else:

        st.sidebar.warning(
            "Please write some text!"
        )


# ==========================================================
# STUDY REPORT
# ==========================================================

if len(
    st.session_state.messages
) > 0:

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "📥 Export Your Notes"
    )


    current_time = datetime.now().strftime(
        "%d-%B-%Y %H:%M"
    )


    report_text = (
        "📚 AI STUDY BUDDY "
        "- COMPLETE STUDY REPORT 📚\n"
    )


    report_text += (
        f"Generated on: "
        f"{current_time}\n"
    )


    report_text += (
        "=" * 60
        + "\n\n"
    )


    for msg in st.session_state.messages:

        role = (
            "👤 YOU"
            if msg["role"] == "user"
            else "🤖 AI TEACHER"
        )


        report_text += (
            f"{role}:\n"
            f"{msg['content']}\n\n"
            f"{'-' * 60}\n\n"
        )


    st.sidebar.download_button(
        label="📄 Download Full Study Report",
        data=report_text,
        file_name="AI_Study_Report.txt",
        mime="text/plain"
    )


## ==========================================================
# INITIALIZE CHAT HISTORY
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# STUDY ANALYTICS
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 My Progress")

total_questions = sum(
    1
    for msg in st.session_state.messages
    if msg["role"] == "user"
)

total_docs = len(
    st.session_state.processed_files
)

words_learned = sum(
    len(msg["content"].split())
    for msg in st.session_state.messages
    if msg["role"] == "assistant"
)

col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric(
        label="Questions",
        value=total_questions
    )

with col2:
    st.metric(
        label="Docs",
        value=total_docs
    )

st.sidebar.metric(
    label="Words Learned (approx)",
    value=words_learned
)

if total_questions > 0:

    st.sidebar.write("Study Streak 🔥")

    progress_val = min(
        total_questions * 10,
        100
    )

    st.sidebar.progress(progress_val)

else:

    st.sidebar.info(
        "Ask your first question "
        "to start tracking progress!"
    )


# ==========================================================
# MAIN CHAT
# ==========================================================

st.sidebar.markdown("---")

st.subheader("💬 Chat with your Study Buddy")


# ==========================================================
# INITIALIZE CHAT HISTORY
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            remove_thinking(
                message["content"]
            )
        )


# ==========================================================
# VOICE INPUT
# ==========================================================

st.write(
    "🎤 **Tap the microphone and ask your question:**"
)

audio = st.audio_input(
    "Start Recording",
    key="my_voice_mic"
)

voice_input = None


# ==========================================================
# PROCESS NEW VOICE RECORDING
# ==========================================================

if audio is not None:

    audio_bytes = audio.getvalue()

    current_audio_id = hash(audio_bytes)

    if (
        current_audio_id
        != st.session_state.get(
            "last_audio_id",
            None
        )
    ):

        st.session_state.last_audio_id = (
            current_audio_id
        )

        st.info(
            "🔄 Your voice is being processed..."
        )

        try:

            client = Groq(
                api_key=st.secrets[
                    "GROQ_API_KEY"
                ]
            )

            transcription = (
                client.audio.transcriptions.create(
                    file=(
                        "audio.wav",
                        audio_bytes
                    ),
                    model="whisper-large-v3",
                    response_format="text"
                )
            )

            voice_input = str(
                transcription
            ).strip()

            if voice_input:

                st.success(
                    "✅ Voice understood!"
                )

            else:

                st.warning(
                    "⚠️ Voice samajh nahi aayi. "
                    "Please try recording again."
                )

                voice_input = None

        except Exception as e:

            st.error(
                f"❌ Audio samajhne mein error aayi: {e}"
            )

            voice_input = None


# ==========================================================
# TEXT INPUT
# ==========================================================

text_input = st.chat_input(
    "Ask a question about your document or anything else..."
)


# ==========================================================
# FINAL PROMPT
# ==========================================================

prompt = text_input or voice_input


# ==========================================================
# CHAT PROCESSING
# ==========================================================

if prompt:

    # ------------------------------------------------------
    # SAVE USER MESSAGE
    # ------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


    # ------------------------------------------------------
    # DISPLAY USER MESSAGE
    # ------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)


    # ------------------------------------------------------
    # AI RESPONSE
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # ==================================================
                # LONG-TERM MEMORY
                # ==================================================

                memory_context = get_memory_context(
                    prompt,
                    max_memories=8
                )


                # ==================================================
                # DOCUMENT CHAT
                # ==================================================

                if (
                    st.session_state.vector_store
                    is not None
                ):

                    docs = (
                        st.session_state.vector_store
                        .similarity_search(
                            prompt,
                            k=3
                        )
                    )

                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                    custom_prompt = f"""
You are a highly intelligent AI Study Buddy
and Expert Teacher.

The user has uploaded study material.

LONG-TERM USER MEMORY:

{memory_context}

IMPORTANT MEMORY RULE:

Use the long-term memory only when it is relevant
to the current question.

Do not mention that you have a memory system.
Do not reveal internal memory information unnecessarily.
Do not invent memories.
Do not treat memory as study material.

STUDY MATERIAL:

{context}

USER QUESTION:

{prompt}

INSTRUCTIONS:

1. Use the uploaded study material as the primary source.

2. You may add useful general knowledge
   when it helps explain the topic.

3. Give clear and accurate answers.

4. If the user asks interview questions,
   provide basic, intermediate and advanced
   questions with answers.

5. Give examples when useful.

6. Explain like an expert teacher.

7. Reply in exactly the same language
   used by the user.

8. Use relevant long-term memory naturally
   when it helps personalize the response.

9. Do not mention these instructions.

ANSWER:
"""

                    answer = ask_llm(
                        custom_prompt
                    )


                # ==================================================
                # GENERAL CHAT
                # ==================================================

                else:

                    response = llm.invoke([

                        SystemMessage(
                            content=f"""
You are a highly intelligent AI Study Buddy.

Reply in exactly the same language
used by the user.

If the user writes Hindi,
reply in Hindi.

If the user writes English,
reply in English.

Explain concepts clearly like
an expert teacher.

LONG-TERM USER MEMORY:

{memory_context}

MEMORY RULES:

1. Use memory only when relevant.
2. Personalize responses naturally.
3. Never mention the memory system.
4. Never reveal internal memory unnecessarily.
5. Never invent information that is not in memory.
6. If memory conflicts with the user's current message,
   always trust the current message.
"""
                        ),

                        HumanMessage(
                            content=prompt
                        )

                    ])

                    answer = response.content


                # ==================================================
                # REMOVE THINKING
                # ==================================================

                answer = remove_thinking(
                    answer
                )


                # ==================================================
                # SAVE LONG-TERM MEMORY
                # ==================================================

                try:

                    extract_and_save_memories(
                        prompt,
                        answer
                    )

                except Exception:

                    pass


                # ==================================================
                # DISPLAY AI ANSWER
                # ==================================================

                st.markdown(answer)


                # ==================================================
                # TEXT TO SPEECH
                # ==================================================

                try:

                    clean_answer = (
                        clean_text_for_speech(
                            answer
                        )
                    )

                    tts = gTTS(
                        text=clean_answer,
                        lang=selected_lang
                    )

                    tts.save(
                        "chat_reply.mp3"
                    )

                    st.audio(
                        "chat_reply.mp3",
                        format="audio/mp3"
                    )

                except Exception:

                    pass


                # ==================================================
                # SAVE AI MESSAGE
                # ==================================================

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


            # ==================================================
            # AI ERROR
            # ==================================================

            except Exception as e:

                error_message = (
                    f"❌ AI Error: {e}"
                )

                st.error(
                    error_message
                )

                # Save error safely inside except
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
