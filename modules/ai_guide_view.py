"""
AI Career Guide - Interactive Chatbot View
Streamlit chat interface powered by local Qwen model and SentenceTransformers RAG.
Features a Gemini/WhatsApp-style layout with right-aligned user messages and left-aligned assistant messages.
"""
import streamlit as st
from ai_engine.chatbot_integration import (
    generate_chatbot_response,
    get_dependencies_status,
)
from modules.ui import page_header


def inject_chat_custom_css():
    """
    Injects clean, minimal CSS for Gemini/WhatsApp-style chat:
    - User messages aligned to the right with a distinct dark bubble
    - Assistant messages aligned to the left with a clean light card
    - Native Streamlit flexbox with proper spacing so avatars never overlap text
    - Full-width aligned chat input
    """
    st.markdown(
        """
        <style>
        /* Container Spacing & Gap */
        div[data-testid="stChatMessage"] {
            gap: 0.75rem !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            background-color: transparent !important;
        }

        /* Prevent Avatars from shrinking */
        div[data-testid="stChatMessageAvatarUser"],
        div[data-testid="stChatMessageAvatarAssistant"] {
            flex-shrink: 0 !important;
        }

        /* -------------------------------------------------------------
           USER MESSAGES (Right-Aligned, Dark Bubble)
        ------------------------------------------------------------- */
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
        div[data-testid="stChatMessage"]:has([aria-label*="user" i]) {
            flex-direction: row-reverse !important;
            background-color: transparent !important;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
        div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] {
            background-color: #1e293b !important;
            color: #ffffff !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 0.8rem 1.15rem !important;
            max-width: 75% !important;
            flex-grow: 0 !important;
            margin-left: auto !important;
            margin-right: 0 !important;
            text-align: left !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] span,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] li,
        div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] span,
        div[data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] li {
            color: #ffffff !important;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) code,
        div[data-testid="stChatMessage"]:has([aria-label*="user" i]) code {
            background-color: rgba(255, 255, 255, 0.2) !important;
            color: #ffffff !important;
        }

        /* -------------------------------------------------------------
           ASSISTANT MESSAGES (Left-Aligned, Light Card)
        ------------------------------------------------------------- */
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
        div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) {
            flex-direction: row !important;
            background-color: transparent !important;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
        div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] {
            background-color: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 0.9rem 1.25rem !important;
            max-width: 82% !important;
            flex-grow: 0 !important;
            margin-right: auto !important;
            margin-left: 0 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] span,
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] li,
        div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] span,
        div[data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] li {
            color: #0f172a !important;
            line-height: 1.65;
        }

        /* -------------------------------------------------------------
           CHAT INPUT
        ------------------------------------------------------------- */
        div[data-testid="stChatInput"] {
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ai_career_guide():
    """Renders the AI Career Guide chat interface with Gemini/WhatsApp style layout."""
    # Inject CSS for right/left message alignment and clean bubbles
    inject_chat_custom_css()

    page_header(
        "🤖 AI Career Guide",
        "Offline Intelligent Assistant for Sri Lankan IT Careers",
        "Ask questions about Sri Lankan tech industry salaries, career progression, required skill stacks, and pathways grounded in our local dataset.",
    )

    status = get_dependencies_status()

    # Show informative notices if any dependencies or data files are pending
    if status["missing_packages"] or not status["dataset_path"]:
        with st.expander("ℹ️ AI Engine Status & Setup Guide", expanded=False):
            if status["missing_packages"]:
                st.warning(
                    f"To enable local LLM inference and full RAG embeddings, install: "
                    f"`pip install {' '.join(status['missing_packages'])}`"
                )
            if not status["dataset_path"]:
                st.warning(
                    "Sri Lankan IT dataset not detected. Please place `srilanka_it_dataset_5000.json` "
                    "in the `data/` directory."
                )
            if not status["adapter_path"]:
                st.info(
                    "Fine-tuned LoRA weights not found at `models/custom_career_lora`. "
                    "Base Qwen or direct knowledge retrieval will be used."
                )

    # Initialize chat session history
    if "ai_guide_messages" not in st.session_state:
        st.session_state.ai_guide_messages = [
            {
                "role": "assistant",
                "content": (
                    "Ayubowan! 🙏 I am your **AI Career Guide**, running completely offline with our "
                    "Sri Lankan IT industry knowledge base and Qwen AI model.\n\n"
                    "Feel free to ask me anything about:\n"
                    "- 💼 **Salaries & Compensation** across Colombo and Sri Lankan tech firms\n"
                    "- 🛠️ **In-demand Skills** for Software Engineers, Data Scientists, and DevOps\n"
                    "- 🎓 **Career Pathways & Guidance** for IT undergraduate students"
                ),
            }
        ]

    # Chat tools row: Clear conversation history
    col_tools, col_space = st.columns([2, 5])
    with col_tools:
        if st.button("🗑️ Clear Chat History", help="Reset conversation history"):
            st.session_state.ai_guide_messages = [
                {
                    "role": "assistant",
                    "content": "Conversation cleared. How can I assist your career journey today?",
                }
            ]
            st.rerun()

    # Display full chat message history using standard Streamlit chat components (no custom avatars)
    for message in st.session_state.ai_guide_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User chat input
    user_prompt = st.chat_input("E.g., What are the typical Software Engineer salaries in Colombo?")

    if user_prompt:
        # Append and render user message immediately
        st.session_state.ai_guide_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate bot response via RAG + Qwen pipeline
        with st.chat_message("assistant"):
            with st.spinner("Consulting Sri Lankan IT dataset and analyzing response..."):
                result = generate_chatbot_response(user_prompt)
                response_text = result["response"]
                st.markdown(response_text)

                # Show explainable context expander if RAG retrieved relevant snippets
                if result.get("context") and result.get("similarity", 0) >= 0.20:
                    with st.expander("🔍 Verified Knowledge Base Evidence", expanded=False):
                        st.caption(f"Semantic match score: {result['similarity']:.3f} | Source: {result['source']}")
                        st.markdown(result["context"])

        # Persist response into chat history
        st.session_state.ai_guide_messages.append(
            {"role": "assistant", "content": response_text}
        )
