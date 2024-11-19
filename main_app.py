"""Este script contiene la aplicación principal."""

import streamlit as st
from src.chunking.chunk import chunk_document

st.title("Hello")


chunk_document(Document(page_content="Hola, ¿cómo estás?"))