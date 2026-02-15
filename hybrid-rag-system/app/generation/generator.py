"""LLM generation layer using LangChain + OpenAI."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.core.logger import logger

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.

Rules:
- Use only information from the context below.
- If the context doesn't contain enough information, say so clearly.
- Be concise and accurate.
- Cite which source document(s) your answer comes from when possible.

Context:
{context}"""

RAG_USER_PROMPT = """Question: {question}

Answer:"""


class RAGGenerator:
    """Generates answers using LangChain with context from retrieval."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=1024,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_USER_PROMPT),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info(f"Generator initialized: model='{settings.openai_model}'")

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer given a query and reranked context."""
        if not context_chunks:
            return "I don't have enough context to answer this question."

        # Build context string from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{chunk['content']}")

        context = "\n\n---\n\n".join(context_parts)

        try:
            answer = self.chain.invoke(
                {"context": context, "question": query}
            )
            logger.info(f"Generated answer: {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating answer: {e}"
