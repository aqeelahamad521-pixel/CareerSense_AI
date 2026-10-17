"""
Chatbot Integration Module for CareerSense AI
Ported from CareerSense_ai: Qwen2.5-1.5B-Instruct + LoRA fine-tuning + SentenceTransformers RAG.
Provides offline, context-grounded AI guidance on Sri Lankan IT careers, salaries, and skills.
"""
from pathlib import Path
import json
import os
import streamlit as st
import numpy as np

# Safe Imports for Optional/Heavy ML Dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TORCH_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    PEFT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent

# Candidate locations for dataset, cache, and LoRA adapter
DATASET_CANDIDATES = [
    BASE_DIR / "data" / "srilanka_it_dataset_5000.json",
    BASE_DIR / "srilanka_it_dataset_5000.json",
    BASE_DIR.parent.parent / "CareerSense_ai" / "srilanka_it_dataset_5000.json",
]

CACHE_CANDIDATES = [
    BASE_DIR / "data" / "embeddings_cache.npy",
    BASE_DIR / "embeddings_cache.npy",
    BASE_DIR.parent.parent / "CareerSense_ai" / "embeddings_cache.npy",
]

ADAPTER_CANDIDATES = [
    BASE_DIR / "models" / "custom_career_lora",
    BASE_DIR / "ai_engine" / "saved_models" / "custom_career_lora",
    BASE_DIR.parent.parent / "CareerSense_ai" / "models" / "custom_career_lora",
]


def resolve_candidate_path(candidates):
    """Finds the first existing path among candidate locations."""
    for path in candidates:
        if path.exists():
            return path
    return None


def get_dependencies_status():
    """Checks the availability of required packages and assets."""
    dataset_path = resolve_candidate_path(DATASET_CANDIDATES)
    cache_path = resolve_candidate_path(CACHE_CANDIDATES)
    adapter_path = resolve_candidate_path(ADAPTER_CANDIDATES)

    missing_packages = []
    if not TORCH_AVAILABLE:
        missing_packages.append("torch")
        missing_packages.append("transformers")
    if not PEFT_AVAILABLE:
        missing_packages.append("peft")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing_packages.append("sentence-transformers")

    return {
        "torch_available": TORCH_AVAILABLE,
        "transformers_available": TORCH_AVAILABLE,
        "peft_available": PEFT_AVAILABLE,
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        "missing_packages": missing_packages,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "cache_path": str(cache_path) if cache_path else None,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "rag_ready": bool(SENTENCE_TRANSFORMERS_AVAILABLE and dataset_path),
        "model_ready": bool(TORCH_AVAILABLE),
    }


@st.cache_resource(show_spinner="Loading Sri Lankan IT Knowledge Base (RAG)...")
def load_rag_index():
    """
    Loads Sri Lankan IT domain dataset and precomputed SentenceTransformer embeddings.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None, None, None

    dataset_path = resolve_candidate_path(DATASET_CANDIDATES)
    if not dataset_path:
        return None, None, None

    cache_path = resolve_candidate_path(CACHE_CANDIDATES)
    if not cache_path:
        # Default save location if generating for the first time
        cache_path = BASE_DIR / "data" / "embeddings_cache.npy"

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = [
            f"{item.get('instruction', '')} {item.get('context', '')} {item.get('response', '')}"
            for item in data
        ]

        model = SentenceTransformer("all-MiniLM-L6-v2")

        compute_embeddings = True
        if cache_path.exists():
            try:
                dataset_mtime = os.path.getmtime(dataset_path)
                cache_mtime = os.path.getmtime(cache_path)
                if dataset_mtime <= cache_mtime:
                    embeddings = np.load(cache_path)
                    if len(embeddings) == len(documents):
                        compute_embeddings = False
            except Exception as e:
                print(f"[CareerSense Chatbot] Cache verification notice: {e}")

        if compute_embeddings:
            print("[CareerSense Chatbot] Computing new embeddings for dataset...")
            embeddings = model.encode(documents, show_progress_bar=False)
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embeddings)
            except Exception as e:
                print(f"[CareerSense Chatbot] Warning saving embeddings cache: {e}")

        return model, embeddings, data
    except Exception as e:
        print(f"[CareerSense Chatbot] Error loading RAG index: {e}")
        return None, None, None


def retrieve_context(query: str, top_k: int = 2):
    """
    Performs semantic retrieval against the Sri Lankan IT dataset.
    Returns (context_string, max_similarity_score).
    """
    rag_model, rag_matrix, rag_data = load_rag_index()
    if rag_model is None or rag_matrix is None or not rag_data:
        return "", 0.0

    try:
        query_vec = rag_model.encode([query])
        similarities = cosine_similarity(query_vec, rag_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0

        context_str = ""
        for idx in top_indices:
            if similarities[idx] > 0.30:  # Semantic similarity threshold
                item = rag_data[idx]
                context_str += f"- {item.get('response', '')}\n"

        return context_str.strip(), max_sim
    except Exception as e:
        print(f"[CareerSense Chatbot] Context retrieval error: {e}")
        return "", 0.0


@st.cache_resource(show_spinner="Loading Local Qwen AI Model & LoRA Adapter...")
def load_local_model():
    """
    Loads base Qwen2.5-1.5B-Instruct model and attaches fine-tuned LoRA weights.
    """
    if not TORCH_AVAILABLE:
        return None, None

    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = resolve_candidate_path(ADAPTER_CANDIDATES)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # Detect optimal device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype
        ).to(device)

        if PEFT_AVAILABLE and adapter_path and adapter_path.exists():
            print(f"[CareerSense Chatbot] Attaching LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            model = base_model

        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"[CareerSense Chatbot] Error loading Qwen model: {e}")
        return None, None


def generate_chatbot_response(prompt: str) -> dict:
    """
    Main pipeline:
    1. Semantic retrieval with SentenceTransformers
    2. Threshold evaluation (0.20 hard-block for out-of-domain)
    3. Context-grounded Qwen generation (with hallucination prevention prompt)
    4. Clean fallback to RAG direct answer if local model is offline.

    Returns dict with keys:
    - 'response': Text message to display
    - 'similarity': Float max similarity score
    - 'context': Retrieved context string
    - 'source': 'qwen_lora' | 'rag_direct' | 'out_of_scope' | 'error'
    """
    retrieved_info, max_sim = retrieve_context(prompt)

    # Hard-block for questions outside Sri Lankan IT scope
    if max_sim < 0.20:
        return {
            "response": "I do not have information regarding that. Please ask questions related to the Sri Lankan IT industry, job roles, salary expectations, or required skills.",
            "similarity": max_sim,
            "context": "",
            "source": "out_of_scope"
        }

    tokenizer, model = load_local_model()

    # If Qwen model is successfully loaded in memory
    if tokenizer is not None and model is not None:
        try:
            if retrieved_info:
                model_prompt = (
                    "System: You are CareerSense AI. You must answer the user's question strictly using "
                    "ONLY the following context. If the answer is not in the context, say 'I don't know'. "
                    "CRITICAL RULE: DO NOT hallucinate, guess, or estimate any numbers, salaries, or statistics. "
                    "If the exact salary figures or numbers are not explicitly mentioned in the provided context, "
                    "you must state that you do not have the specific data. "
                    "CRITICAL RULE: Be highly concise. Provide short, direct answers. Do not write long essays, extensive lists, or unnecessary filler text.\n\n"
                    f"Context: {retrieved_info}\n\n"
                    f"User: {prompt}\nAssistant:"
                )
            else:
                model_prompt = f"User: {prompt}\nAssistant:"

            inputs = tokenizer(model_prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_reply = decoded.split("Assistant:")[-1].strip()

            return {
                "response": bot_reply,
                "similarity": max_sim,
                "context": retrieved_info,
                "source": "qwen_lora"
            }
        except Exception as e:
            print(f"[CareerSense Chatbot] Model inference error: {e}")
            # Fall through to RAG direct response below

    # Fallback: If Qwen is still loading or running on lightweight CPU without torch weights
    if retrieved_info:
        clean_context = retrieved_info.replace("- ", "• ").strip()
        fallback_reply = (
            f"Here is what our Sri Lankan IT career knowledge base indicates for your query:\n\n"
            f"{clean_context}\n\n"
            f"*(Generated from local dataset retrieval, similarity: {max_sim:.2f})*"
        )
        return {
            "response": fallback_reply,
            "similarity": max_sim,
            "context": retrieved_info,
            "source": "rag_direct"
        }

    return {
        "response": "I encountered an issue generating a response. Please verify that the model weights and dataset are installed.",
        "similarity": max_sim,
        "context": "",
        "source": "error"
    }
