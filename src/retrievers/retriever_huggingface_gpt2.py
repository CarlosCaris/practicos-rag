from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from transformers import pipeline

from tqdm import tqdm
import random

class RAGPipeline:
    def __init__(self, qdrant_vector_store):
        self.qdrant_vector_store = qdrant_vector_store
        self.retriever = self.qdrant_vector_store

        self.llm = pipeline("text-generation", model="gpt2") 
        self.prompt = self._create_prompt_template()
        self.qa_generation_prompt = self._create_qa_prompt_template()
        self.rag_chain = self._setup_rag_chain()
        self.question_chain = self._setup_question_chain()

    def _create_prompt_template(self):
        template = """Utilize the retrieved context below to answer the question.
        If you're unsure of the answer, simply state you don't know and apologize.
        Keep your response concise, limited to two sentences.
        Question: {question}
        Context: {context}
        """
        return ChatPromptTemplate.from_template(template)

    def _create_qa_prompt_template(self):
        template = """Your task is to write a factoid question and an answer given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

        Provide your answer as follows:

        Output:::
        Factoid question: (your factoid question)
        Answer: (your answer to the factoid question)

        Now here is the context.

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
