from rerankers import Reranker

def advanced_retrieval_reranking(session_state, query, questions_processed, ground_truths):
    """
    Sistema avanzado de retrieval con reranking
    """
    try:
        contexts = []
        answers = []
        
        for question in questions_processed:
            # Obtener respuesta usando el chain
            answer = session_state.obj_retriever.rag_chain.invoke(question)
            answers.append(answer if answer else "Sin respuesta")
            
            # # Obtener documentos iniciales
            # retrieved_docs = st.session_state.obj_retriever.retriever.get_relevant_documents(
            #     question,
            #     kwargs={"k": 10}  # Recuperamos más documentos inicialmente para el reranking
            # )
            
            # Aplicar reranking
            #reranked_results = rerank_docs(question, retrieved_docs)
            reranked_results = rerank_docs(session_state, query)
            
            # Tomar los mejores documentos reranqueados
            if reranked_results and hasattr(reranked_results, 'results') and reranked_results.results:
                # Si el reranker devuelve resultados en formato específico
                top_docs = [result.document.text for result in reranked_results.results[:3]]
                contexts.append(top_docs)
            else:
                # Si el reranker devuelve una lista simple
                contexts.append([doc for doc in reranked_results[:3]] if reranked_results else ["Sin contexto relevante"])


        # if "advanced_df" not in session_state:
        #     # Crear un DataFrame vacío o con datos iniciales
        return questions_processed, answers, contexts, ground_truths
    
    except Exception as e:
        print(f"Error durante la búsqueda: {e}")


def rerank_docs(session_state, query):
    """
    Reranquea los documentos usando el modelo cross-encoder
    """
    sampled_docs = session_state.obj_retriever.retriever.get_relevant_documents(query)
    sampled_docs_processed = [doc.page_content for doc in sampled_docs]
    reranker = Reranker("cross-encoder", verbose=0, model_type='cross-encoder')
    reranked_docs = reranker.rank(query, sampled_docs_processed)
    return reranked_docs