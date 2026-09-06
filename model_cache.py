# model_cache.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
from gemini_client import gemini


@st.cache_resource
def load_embedding_model():
    """
    Loads sentence embedding model.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_generator():
    """
    Returns Gemini generator.
    """
    return gemini


@st.cache_resource
def load_summarizer():
    """
    Returns Gemini summarizer.
    """
    return gemini