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
        results = self.index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        chunks = []
        sources = []
        for match in results["matches"]:
            print(f"Score: {match['score']} | Source: {match['metadata'].get('source')}")
            if match["score"] > 0.1:
                chunks.append(match["metadata"].get("text", ""))
                sources.append(match["metadata"].get("source", "Unknown"))

        if not chunks:
            return {
                "answer": "I could not find relevant information to answer your question.",
                "sources": []
            }

        context = "\n\n".join(chunks)
        prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

        response = self.groq.chat.completions.create(
            model="llama
