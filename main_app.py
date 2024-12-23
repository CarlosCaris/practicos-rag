import streamlit as st
from rerankers import Reranker
import json
import random
from settings import MODEL_NAME, DATA_PATH, SAMPLE_N, MAX_LENGTH, MIN_LENGTH
from src.vector_store_client.qdrant_store_openai import QdrantManager
from src.retrievers.retriever_openai import RAGPipeline
from src.reranking.reranking_manager import advanced_retrieval_reranking
from src.loaders.load_documents import fn_load_pdf
from transformers import pipeline
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Inicialización de estados
def initialize_session_state(key, initializer):
    if key not in st.session_state:
        st.session_state[key] = initializer()

def initialize_vector_store(model_name):
    if "vector_store_initialized" not in st.session_state:
        with st.spinner("Cargando base de datos y retriever..."):
            st.session_state.obj_vector_store = QdrantManager()
            st.session_state.obj_vector_store.load_data()
            st.session_state.obj_retriever = RAGPipeline(
                st.session_state.obj_vector_store.retriever, model_name
            )
            st.session_state.vector_store_initialized = True
        st.success("Base de datos y Retriever cargados exitosamente.")

def initialize_document_loader(data_path):
    if "documents_loaded" not in st.session_state:
        with st.spinner("Cargando documentos..."):
            st.session_state.documents = fn_load_pdf(data_path)
            st.session_state.documents_loaded = True

def display_mean_results(df1, df2):
    st.markdown("## Métricas Promedio: Baseline vs Advanced retrieval.")
    st.table({
        "Métrica": ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"],
        "Promedio Baseline": [
            round(df1["faithfulness"].mean(), 4),
            round(df1["answer_relevancy"].mean(), 4),
            round(df1["context_recall"].mean(), 4),
            round(df1["context_precision"].mean(), 4),
        ],
        "Promedio Advanced": [
            round(df2["faithfulness"].mean(), 4),
            round(df2["answer_relevancy"].mean(), 4),
            round(df2["context_recall"].mean(), 4),
            round(df2["context_precision"].mean(), 4),
        ],
    })

def display_results(questions, answers, contexts, ground_truths):
    st.write("### Preguntas y respuestas generadas")
    for i in range(len(questions)):
        with st.expander(f"Resultado {i + 1}:"):
            st.write(f"**Pregunta:** {questions[i]}")
            st.write(f"**Respuesta Generada:** {answers[i]}")
            st.write(f"**Contexto Recuperado:**")
            for context in contexts[i]:
                st.write(f"- {context}")
            st.write(f"**Respuesta de Referencia:** {ground_truths[i]}")

    data = {
        "question": questions,
        "answer": answers,
        "reference": ground_truths,
        "retrieved_contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    evaluate_result = evaluate(
        dataset=dataset,
        llm=st.session_state.obj_retriever.llm,
        embeddings=st.session_state.obj_vector_store.embedding_model,
        metrics=[
            context_recall,
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )

    evaluate_result_df = evaluate_result.to_pandas()
    st.markdown("### Métricas de Evaluación")
    st.dataframe(evaluate_result_df)

    return evaluate_result_df

def show_chunking_results_page():
    st.title("Resultados de Limpieza y Chunking al azar.")
    with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
                st.session_state["processed_chunks"] = json.load(f)
    num_chunks = st.number_input("Número de fragmentos a mostrar", min_value=1, max_value=100, value=5, step=1)
    selected_chunks = random.sample(st.session_state["processed_chunks"], num_chunks)
    for i, chunk in enumerate(selected_chunks, start=1):
        with st.expander(f"Fragmento {i}..."):
            st.write(f"**Contenido:** {chunk['page_content']}")
            st.write(f"**Metadatos:** {chunk['metadata']}")

def prepare_questions_and_ground_truths(query, sample_n=10):
    """
    Prepara las preguntas y ground truths que serán usados por ambos sistemas.
    """
    questions = st.session_state.obj_retriever.batch_generate_factoid_questions(
        st.session_state.documents, sample_n=sample_n
    )
    
    questions_processed = []
    ground_truths = []
    
    for question in questions:
        if "Factoid question: " in question and "Answer: " in question:
            question_processed = question.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
            ground_truth = question.split("Answer: ")[-1].strip()
            
            if not question_processed or not ground_truth:
                continue
                
            questions_processed.append(question_processed)
            ground_truths.append(ground_truth)
    
    return questions_processed, ground_truths

def baseline_retrieval(questions_processed, ground_truths):
    """
    Sistema baseline de retrieval sin rerank
    """
    contexts = []
    answers = []
    
    for question in questions_processed:
        answer = st.session_state.obj_retriever.rag_chain.invoke(question)
        answers.append(answer if answer else "Sin respuesta")
        
        context = [
            docs.page_content for docs in st.session_state.obj_retriever.retriever.get_relevant_documents(question)
        ]
        contexts.append(context if context else ["Sin contexto relevante"])

    # Crear un DataFrame vacío o con datos iniciales
    st.session_state["baseline_df"] = display_results(questions_processed, answers, contexts, ground_truths)

def generate_summary(contexts, max_length=MAX_LENGTH, min_length=MIN_LENGTH):
    
    """
    Genera un resumen de una lista de contextos recuperados.

    Args:
    contexts (list): Lista de textos recuperados.
    max_length (int): Longitud máxima del resumen.
    min_length (int): Longitud mínima del resumen.

    Returns:
    str: Resumen generado.
    """
    # Inicializar el modelo de resumen
    summarizer = pipeline("summarization")
    combined_text = " ".join(contexts)
    if not combined_text.strip():
        return "No se pudo generar un resumen debido a la falta de contenido relevante."
    summary = summarizer(combined_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

def main():
    load_dotenv()
    initialize_vector_store(MODEL_NAME)
    initialize_document_loader(DATA_PATH)

    option = st.sidebar.selectbox("Selecciona una opción:", ["Inicio", "Chunking", "Consulta"])

    if option == "Inicio":
        st.title("Bienvenido al sistema RAG para regulaciones de alimentos en Chile.")
        st.markdown("### Descripción de la Aplicación")
        st.markdown(
            """
            Esta aplicación permite realizar búsquedas avanzadas en documentos relacionados con las regulaciones de alimentos en Chile. 
            Utiliza técnicas de Recuperación Aumentada de Generación (RAG) para generar respuestas basadas en documentos procesados y fragmentados. 

            ### Cómo Funciona:
            1. **Preprocesamiento (Paso Inicial)**: Antes de realizar consultas, es necesario procesar y fragmentar los documentos cargados para optimizar la precisión y relevancia de las respuestas.
            2. **Chunking**: Visualiza los fragmentos generados durante el preprocesamiento.
            3. **Consulta**: Puedes realizar preguntas específicas sobre las regulaciones y obtener respuestas generadas basadas en los documentos procesados.

            Utiliza el menú de navegación en el lado izquierdo para acceder a estas funcionalidades.
            """
        )

    elif option == "Consulta":
        st.title("Evaluador de RAG para regulaciones de alimentos en Chile.")

        # Manejo de la consulta
        query = st.text_input("Ingresa tu consulta:")

        if st.button("Buscar") and query.strip():
            questions_processed, ground_truths = prepare_questions_and_ground_truths(query, SAMPLE_N)

            st.markdown("## Baseline Retrieval")
            baseline_retrieval(questions_processed, ground_truths)

            st.markdown("## Advanced Retrieval with Rerank")
                        
            # Crear un DataFrame vacío o con datos iniciales
            questions_processed, answers, contexts, ground_truths = advanced_retrieval_reranking(st.session_state, query, questions_processed, ground_truths)
            st.session_state["advanced_df"] = display_results(questions_processed, answers, contexts, ground_truths)

            display_mean_results(st.session_state.baseline_df, st.session_state.advanced_df)

    elif option == "Chunking":
        show_chunking_results_page()

if __name__ == "__main__":
    main()