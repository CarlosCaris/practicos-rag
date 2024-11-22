import streamlit as st
from src.vector_store_client.qdrant_store import QdrantManager

def main():
    st.title("Búsqueda Semántica en Documentos PDF")

    # Inicializar QdrantManager
    qdrant_manager = QdrantManager()
    qdrant_manager.load_data()

    # Input de consulta
    query = st.text_input("Ingresa tu consulta:")

    # Botón de búsqueda
    if st.button("Buscar"):
        if not query.strip():
            st.warning("Por favor, ingresa una consulta válida.")
        else:
            results = qdrant_manager.dense_search(query, top_k=5)

            # Mostrar resultados
            if results:
                st.write("Resultados:")
                for i, result in enumerate(results):
                    st.markdown(f"### Resultado {i + 1}")
                    st.write(result.page_content)
                    st.write(f"**Metadatos:** {result.metadata}")
            else:
                st.write("No se encontraron resultados.")

if __name__ == "__main__":
    main()
