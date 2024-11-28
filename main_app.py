import streamlit as st
from src.vector_store_client.qdrant_store import QdrantManager

def initialize_qdrant():
    """
    Inicializa QdrantManager y carga la base de datos si no está en session_state.
    """
    if "qdrant_manager" not in st.session_state:
        st.session_state.qdrant_manager = QdrantManager()
        st.session_state.qdrant_manager.load_data()
        st.success("Base de datos cargada exitosamente.")

def main():
    st.title("Búsqueda Semántica en Documentos PDF")

    # Inicializar QdrantManager una sola vez
    initialize_qdrant()

    # Input de consulta
    query = st.text_input("Ingresa tu consulta:")

    # Botón de búsqueda
    if st.button("Buscar"):
        if not query.strip():
            st.warning("Por favor, ingresa una consulta válida.")
        else:
            try:
                # Realizar la búsqueda densa con puntajes
                results_with_scores = st.session_state.qdrant_manager.search(query, top_k=5)

                # Mostrar resultados
                if results_with_scores:
                    st.write("Resultados ordenados por relevancia (fragmentos relacionados):")
                    for i, (doc, score) in enumerate(results_with_scores):
                        st.markdown(f"### Chunk {i + 1}")
                        st.write(doc.page_content)  # Muestra el chunk completo
                        st.write(f"**Puntaje de similitud:** {score:.4f}")
                        st.write(f"**Metadatos:** {doc.metadata}")
                else:
                    st.write("No se encontraron resultados.")
            except Exception as e:
                st.error(f"Error durante la búsqueda: {e}")

if __name__ == "__main__":
    main()
