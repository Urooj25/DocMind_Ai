from pinecone import Pinecone
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    def __init__(self):
        print("Connecting to Pinecone & using Cloud Inference...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))

        print("Connecting to Groq...")
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        print("RAG Engine ready!")

    def get_embedding(self, text: str, input_type: str = "query"):
        response = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[text],
            parameters={"input_type": input_type}
        )
        return response.data[0].values

    def answer(self, question: str) -> dict:
        query_vector = self.get_embedding(question)
