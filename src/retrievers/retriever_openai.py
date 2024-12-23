from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI

from tqdm import tqdm
import random

class RAGPipeline:
    def __init__(self, retriever, llm_model):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=llm_model) 
        self.prompt = self._create_prompt_template()
        self.qa_generation_prompt = self._create_qa_prompt_template()
        self.rag_chain = self._setup_rag_chain()
        self.question_chain = self._setup_question_chain()

    def _create_prompt_template(self):
        template = """Usa el contexto recuperado a continuación para responder la pregunta de manera precisa.
        Si la respuesta no puede determinarse a partir del contexto, indica que no lo sabes y discúlpate.
        Sé claro y proporciona una respuesta relevante relacionada con las regulaciones alimentarias en Chile.
        Question: {question}
        Context: {context}
        """
        return ChatPromptTemplate.from_template(template)

    def _create_qa_prompt_template(self):
        template = """Tu tarea es escribir una pregunta factual y una respuesta dada un contexto.  
        La pregunta factual debe ser respondida con una información específica y concisa tomada directamente del contexto.  
        La pregunta factual debe formularse en el mismo estilo que las preguntas que los usuarios podrían hacer en un motor de búsqueda.  
        Esto significa que la pregunta factual NO DEBE mencionar expresiones como "según el pasaje" o "contexto".  

        Proporciona tu respuesta de la siguiente manera:  

        Output:::  
        Factoid question: (tu pregunta factual)  
        Answer: (tu respuesta a la pregunta factual)  

        Ahora aquí está el contexto.  

        Context: {context}  
        Output:::  
        """

        return ChatPromptTemplate.from_template(template)

    def _setup_rag_chain(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm
            | StrOutputParser() 
        )

    def _setup_question_chain(self):
        return (
            {"context": RunnablePassthrough()}
            | self.qa_generation_prompt
            | self.llm
            | StrOutputParser()
        )

    def run_rag_chain(self, question):
        return self.rag_chain.invoke({"question": question})

    def generate_factoid_question(self, context):
        return self.question_chain.invoke({"context": context})

    def batch_generate_factoid_questions(self, docs, sample_n):
        sampled_docs = random.sample(docs, sample_n)
        sampled_docs_processed = [doc.page_content for doc in sampled_docs]

        questions = [self.question_chain.invoke({"context": sampled_context}) for sampled_context in tqdm(sampled_docs_processed)]
        return questions

    def batch_generate_factoid_questions_with_relevant_docs(self, query, sample_n):
        sampled_docs = self.retriever.retriever.get_relevant_documents(query)
        sampled_docs_processed = [doc.page_content for doc in sampled_docs]

        questions = [self.question_chain.invoke({"context": sampled_context}) for sampled_context in tqdm(sampled_docs_processed)]
        return questions
