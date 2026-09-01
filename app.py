import os
import json
import base64
from datetime import datetime

import streamlit as st


st.markdown("""
<style>

/* Hide ONLY App Creator Avatar (GitHub icon) */
a[aria-label="App Creator Avatar"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain

from gtts import gTTS
#from streamlit_mic_recorder import speech_to_text
from streamlit_mic_recorder import mic_recorder
from groq import Groq
from gtts import gTTS
import streamlit.components.v1 as components



# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide"
)

st.set_option("client.toolbarMode", "viewer")


## ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
   BUTTONS
========================================================== */

.stButton>button {
    transition: all 0.3s ease;
    border-radius: 8px;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    border-color: #4CAF50;
}


/* ==========================================================
   CHAT
========================================================== */

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
}


/* ==========================================================
   MOBILE SCROLLING
========================================================== */

html {
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior-y: none !important;
}

body {
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior-y: none !important;
    -webkit-overflow-scrolling: touch !important;
}


/* Main Streamlit app */

[data-testid="stAppViewContainer"] {
    overflow-x: hidden !important;
    overscroll-behavior-y: none !important;
}


/* Main content */

[data-testid="stMain"] {
    overflow-x: hidden !important;
}


/* Sidebar */

[data-testid="stSidebar"] {
    overflow-x: hidden !important;
}


/* Sidebar content */

[data-testid="stSidebarContent"] {
    overflow-x: hidden !important;
}


/* Prevent components from creating horizontal scroll */

iframe {
    max-width: 100% !important;
    border: none !important;
}


/* ==========================================================
   HIDE STREAMLIT BADGES
========================================================== */

[class*="viewerBadge"],
[class*="styles_viewerBadge"],
[data-testid="stAppDeployButton"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)
# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ==========================================================
# GROQ TEXT MODEL
# ==========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="openai/gpt-oss-20b",
        temperature=0
    )


llm = get_llm()
# ==========================================================
# GROQ CLIENT FOR SPEECH-TO-TEXT
# ==========================================================

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ==========================================================
# HELPER: CLEAN AI JSON
# ==========================================================

def clean_json_response(text):

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]

    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    text = text.strip()

    # Remove accidental text before JSON
    first_bracket = text.find("[")

    if first_bracket != -1:
        text = text[first_bracket:]

    last_bracket = text.rfind("]")

    if last_bracket != -1:
        text = text[:last_bracket + 1]

    return text.strip()


# ==========================================================
# HELPER: TEXT RESPONSE
# ==========================================================

def ask_llm(prompt):

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])

    return response.content


# ==========================================================
# TITLE
# ==========================================================

st.title("📚 AI Study Buddy")

st.write(
    "Upload your study material and search concepts instantly!"
)


# ==========================================================
# SIDEBAR - UPLOAD
# ==========================================================

st.sidebar.header("📁 Upload Study Material")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs or Images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)


# ==========================================================
# PDF + IMAGE PROCESSING
# ==========================================================

if uploaded_files:

    pdf_files = [
        f for f in uploaded_files
        if f.type == "application/pdf"
    ]

    image_files = [
        f for f in uploaded_files
        if "image" in f.type
    ]


    # ======================================================
    # PDF PROCESSING
    # ======================================================

    if pdf_files:

        current_pdf_names = [
            file.name for file in pdf_files
        ]

        if current_pdf_names != st.session_state.processed_files:

            with st.spinner(
                "⚙️ Auto-processing your new PDFs..."
            ):

                all_documents = []

                os.makedirs(
                    "data/uploaded_pdfs",
                    exist_ok=True
                )

                for uploaded_file in pdf_files:

                    file_path = os.path.join(
                        "data/uploaded_pdfs",
                        uploaded_file.name
                    )

                    with open(file_path, "wb") as f:
                        f.write(
                            uploaded_file.getbuffer()
                        )

                    loader = PyPDFLoader(file_path)

                    docs = loader.load()

                    all_documents.extend(docs)


                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                splits = text_splitter.split_documents(
                    all_documents
                )


                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )


                # FAISS
                vector_store = FAISS.from_documents(
                    splits,
                    embeddings
                )

                st.session_state.vector_store = vector_store

                vector_store.save_local(
                    "faiss_index"
                )

                st.session_state.processed_files = (
                    current_pdf_names
                )


            st.sidebar.success(
                f"✅ Automatically processed "
                f"{len(pdf_files)} PDF(s)!"
            )

        else:

            st.sidebar.success(
                f"✅ {len(pdf_files)} PDF(s) ready for chat!"
            )


    # ======================================================
    # IMAGE PROCESSING
    # ======================================================

    if image_files:

        st.sidebar.markdown("---")

        st.sidebar.subheader("📸 Uploaded Image")

        img_to_process = image_files[0]

        st.sidebar.image(
            img_to_process,
            caption=img_to_process.name,
            use_container_width=True
        )


        if st.sidebar.button("🔍 Analyze Image"):

            with st.spinner(
                "AI is looking at your image..."
            ):

                try:

                    image_bytes = img_to_process.getvalue()

                    image_base64 = base64.b64encode(
                        image_bytes
                    ).decode("utf-8")


                    # Existing vision model
                    vision_llm = ChatGroq(
                        api_key=st.secrets["GROQ_API_KEY"],
                        model_name="qwen/qwen3.6-27b"
                    )


                    # Detect MIME type
                    mime_type = img_to_process.type

                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": """
You are an expert teacher.

Analyze this image carefully.

Explain:
1. What is shown in the image?
2. Important concepts
3. Important definitions
4. Examples if useful
5. Exam-important points

Give the explanation in a student-friendly way.
"""
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


                    st.session_state.messages.append({
                        "role": "user",
                        "content":
                        f"📸 User uploaded image: "
                        f"{img_to_process.name}"
                    })


                    st.session_state.messages.append({
                        "role": "assistant",
                        "content":
                        f"**Image Analysis:**\n\n"
                        f"{response.content}"
                    })


                    st.rerun()


                except Exception as e:

                    st.sidebar.error(
                        f"Error analyzing image: {e}"
                    )


# ==========================================================
# SMART STUDY TOOLS
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Smart Study Tools")


# ==========================================================
# QUIZ
# ==========================================================

num_questions = st.sidebar.slider(
    "How many questions?",
    min_value=1,
    max_value=100,
    value=5
)


if st.sidebar.button("📝 Generate MCQ Quiz"):

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

                clean_text = clean_json_response(
                    raw_text
                )

                quiz_data = json.loads(
                    clean_text
                )


                if not isinstance(
                    quiz_data,
                    list
                ):
                    raise ValueError(
                        "Quiz response is not a list"
                    )


                st.session_state.quiz_data = (
                    quiz_data
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
            "Please upload and process a PDF first!"
        )


# ==========================================================
# SUMMARY
# ==========================================================

if st.sidebar.button("📄 Generate Summary"):

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
            "Please upload and process a PDF first!"
        )



# ==========================================================
# MIND MAP
# ==========================================================

if st.sidebar.button("🗺️ Generate Mind Map"):

    if st.session_state.vector_store is not None:

        with st.spinner("Designing your Mind Map..."):

            try:

                # --------------------------------------------------
                # GET RELEVANT STUDY MATERIAL
                # --------------------------------------------------

                docs = st.session_state.vector_store.similarity_search(
                    "core concepts structure important topics",
                    k=8
                )

                context = "\n\n".join(
                    str(doc.page_content)
                    for doc in docs
                )

                # --------------------------------------------------
                # MIND MAP PROMPT
                # --------------------------------------------------

                mindmap_prompt = f"""
You are an expert educational mind-map designer.

Create a simple VALID Graphviz DOT mind map
from the study material below.

MATERIAL:
{context}

STRICT RULES:

1. Return ONLY Graphviz DOT code.
2. Do NOT use markdown.
3. Do NOT use ``` or code fences.
4. Do NOT write explanations.
5. Start with:
digraph G {{
6. End with:
}}
7. Use rankdir=LR.
8. Use a central topic with major topics and subtopics.
9. Every node label must be inside double quotes.
10. Every edge must use:
"Parent" -> "Child";
11. Avoid quotation marks inside node labels.
12. Avoid special characters inside node labels.
13. Keep the mind map simple.
14. Maximum 15-20 nodes.

Use this style:

digraph G {{
    rankdir=LR;

    node [
        shape=box,
        style="filled,rounded",
        fillcolor="#E8F5E9",
        color="#4CAF50",
        fontname="Arial",
        fontsize=20,
        fontcolor="#1B5E20"
    ];

    edge [
        color="#9E9E9E",
        penwidth=2
    ];

    "Central Topic"
        [fillcolor="#FFF9C4",
         color="#FBC02D",
         fontsize=26];

    "Central Topic" -> "Major Topic 1";
    "Major Topic 1" -> "Sub Topic 1";
    "Major Topic 1" -> "Sub Topic 2";
}}
"""

                # --------------------------------------------------
                # CALL AI
                # --------------------------------------------------

                raw_response = ask_llm(mindmap_prompt)

                if raw_response is None:
                    raise ValueError(
                        "AI returned an empty response."
                    )

                # --------------------------------------------------
                # HANDLE DIFFERENT AI RESPONSE FORMATS
                # --------------------------------------------------

                if isinstance(raw_response, dict):

                    raw_text = (
                        raw_response.get("content")
                        or raw_response.get("text")
                        or raw_response.get("response")
                        or raw_response.get("message")
                        or ""
                    )

                elif isinstance(raw_response, list):

                    raw_text = " ".join(
                        str(x) for x in raw_response
                    )

                else:

                    raw_text = str(raw_response)

                raw_text = raw_text.strip()

                if not raw_text:
                    raise ValueError(
                        "AI returned an empty response."
                    )

                # --------------------------------------------------
                # CLEAN RESPONSE
                # --------------------------------------------------

                # Remove common markdown fences
                raw_text = raw_text.replace(
                    "```graphviz", ""
                )

                raw_text = raw_text.replace(
                    "```dot", ""
                )

                raw_text = raw_text.replace(
                    "```Graphviz", ""
                )

                raw_text = raw_text.replace(
                    "```", ""
                )

                raw_text = raw_text.strip()

                # --------------------------------------------------
                # EXTRACT GRAPHVIZ CODE
                # --------------------------------------------------

                start_idx = raw_text.lower().find("digraph")

                if start_idx == -1:

                    raise ValueError(
                        "AI response did not contain Graphviz DOT code."
                    )

                # Take everything starting from digraph
                clean_dot = raw_text[start_idx:].strip()

                # Find the last closing brace
                end_idx = clean_dot.rfind("}")

                if end_idx == -1:

                    raise ValueError(
                        "Graphviz code has no closing bracket."
                    )

                clean_dot = clean_dot[:end_idx + 1].strip()

                # --------------------------------------------------
                # BASIC VALIDATION
                # --------------------------------------------------

                if not clean_dot.lower().startswith("digraph"):

                    raise ValueError(
                        "Invalid Graphviz format."
                    )

                if not clean_dot.endswith("}"):

                    raise ValueError(
                        "Graphviz code is incomplete."
                    )

                # --------------------------------------------------
                # DISPLAY
                # --------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "🗺️ Your Concept Map"
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
            "Please upload and process a PDF first!"
        )# ==========================================================
# 🗂️ GENERATE FLASHCARDS
# ==========================================================

if st.sidebar.button("🗂️ Generate Flashcards"):

    if "vector_store" not in st.session_state:
        st.sidebar.error("Please upload a PDF first!")

    else:
        with st.spinner("Creating Flashcards..."):

            try:
                # Get PDF content
                docs = st.session_state.vector_store.similarity_search(
                    "important terms definitions key concepts",
                    k=5
                )

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                # Create AI prompt
                prompt = f"""
Create exactly 6 important flashcards from this study material:

{context}

Return ONLY valid JSON in this format:

[
  {{
    "term": "Term",
    "definition": "Simple definition"
  }}
]

No markdown. No explanation.
"""

                # Ask Groq
                response = llm.invoke([
                    HumanMessage(content=prompt)
                ])

                text = response.content.strip()

                # Remove markdown if AI adds it
                text = text.replace("```json", "")
                text = text.replace("```", "").strip()

                # Extract JSON
                start = text.find("[")
                end = text.rfind("]") + 1

                cards = json.loads(text[start:end])

                # Save cards
                st.session_state.flashcards = cards[:6]

                st.sidebar.success("✅ Flashcards ready!")

                st.rerun()

            except Exception as e:
                st.sidebar.error(
                    f"Flashcard error: {e}"
                )


# ==========================================================
# 🗂️ DISPLAY FLASHCARDS
# ==========================================================

if "flashcards" in st.session_state:

    st.markdown("---")
    st.subheader("🗂️ Interactive Flashcards")
    st.caption("Hover over a card to flip it 👆")

    cols = st.columns(3)

    for i, card in enumerate(st.session_state.flashcards):

        term = str(card.get("term", ""))
        definition = str(card.get("definition", ""))

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
                    {term}
                </div>

                <div class="flip-card-back">
                    {definition}
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


    # ------------------------------------------------------
# QUIZ DISPLAY
# ==========================================================

if "quiz_data" in st.session_state:

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


            user_answers.append(ans)

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


        if st.button("❌ Close Quiz"):

            del st.session_state.quiz_data

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
    value=16
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
# LANGUAGE
# ==========================================================

lang_option = st.sidebar.selectbox(
    "🗣️ Choose AI Voice Language:",
    ["English", "Hindi"]
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
    "🗑️ Clear Chat History"
):

    st.session_state.messages = []

    st.rerun()


# ==========================================================
# TEXT TO SPEECH SIDEBAR
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.title(
    "Text to Speech App"
)


sidebar_text = st.sidebar.text_area(
    "Write your text here:",
    ""
)


if st.sidebar.button(
    "Play Audio"
):

    if sidebar_text.strip() != "":

        try:

            tts = gTTS(
                text=sidebar_text,
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

if (
    "messages" in st.session_state
    and len(st.session_state.messages) > 0
):

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


# ==========================================================
# STUDY ANALYTICS
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📊 My Progress"
)


total_questions = sum(
    1
    for msg in st.session_state.get(
        "messages",
        []
    )
    if msg["role"] == "user"
)


total_docs = len(
    st.session_state.get(
        "processed_files",
        []
    )
)


words_learned = sum(
    len(msg["content"].split())
    for msg in st.session_state.get(
        "messages",
        []
    )
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

    st.sidebar.write(
        "Study Streak 🔥"
    )

    progress_val = min(
        total_questions * 10,
        100
    )

    st.sidebar.progress(
        progress_val
    )

else:

    st.sidebar.info(
        "Ask your first question "
        "to start tracking progress!"
    )


# ==========================================================
# MAIN CHAT
# ==========================================================

st.sidebar.markdown("---")

st.subheader(
    "💬 Chat with your Study Buddy"
)


# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )




# ==========================================================
# 🎤 VOICE INPUT
# ==========================================================

st.write("🎤 **Tap the microphone and speak:**")

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
    format="webm",
    key="voice_recorder"
)

voice_input = None


if audio is not None:

    try:

        # --------------------------------------------------
        # Send recorded audio to Groq Whisper
        # --------------------------------------------------

        transcription = groq_client.audio.transcriptions.create(
            file=(
                "voice.webm",
                audio["bytes"]
            ),
            model="whisper-large-v3-turbo",
            language="en",
            response_format="json",
            temperature=0
        )

        voice_input = transcription.text.strip()

        if voice_input:

            st.success(
                f"🎤 You said: {voice_input}"
            )

    except Exception as e:

        st.error(
            f"Voice input error: {e}"
        )


# ==========================================================
# TEXT INPUT
# ==========================================================

text_input = st.chat_input(
    "Ask a question about your PDF or anything else..."
)


# Voice OR Text
prompt = text_input or voice_input

# ==========================================================
# CHAT PROCESSING
# ==========================================================

if prompt:

    # ------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })


    with st.chat_message("user"):

        st.markdown(prompt)


    # ------------------------------------------------------
    # AI RESPONSE
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                # ==================================================
                # PDF CHAT
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

STUDY MATERIAL:

{context}

USER QUESTION:

{prompt}

INSTRUCTIONS:

1. Use the uploaded study material as the
   primary source.

2. You may add useful general knowledge
   when it helps explain the topic.

3. Give clear and accurate answers.

4. If the user asks interview questions,
   provide basic, intermediate and advanced
   questions with answers.

5. Give examples when useful.

6. Explain like an expert teacher.

7. IMPORTANT:
   Reply in exactly the same language
   used by the user.

8. Do not mention these instructions.

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
                            content="""
You are a highly intelligent AI Study Buddy.

Reply in exactly the same language
used by the user.

If the user writes Hindi,
reply in Hindi.

If the user writes English,
reply in English.

Explain concepts clearly like
an expert teacher.
"""
                        ),

                        HumanMessage(
                            content=prompt
                        )

                    ])


                    answer = response.content


                # --------------------------------------------------
                # DISPLAY ANSWER
                # --------------------------------------------------

                st.markdown(
                    answer
                )


                # --------------------------------------------------
                # TEXT TO SPEECH
                # --------------------------------------------------

                try:

                    tts = gTTS(
                        text=answer,
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


                # --------------------------------------------------
                # SAVE ANSWER
                # --------------------------------------------------

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


            except Exception as e:

                error_message = (
                    f"❌ AI Error: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })