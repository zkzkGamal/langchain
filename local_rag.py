from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import logging
from vector import retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up local RAG with Ollama and Chroma")

model_name = "qwen3-vl:4b-instruct-q4_K_M"

logger.info("Initializing ChatOllama model")
chat_ollama_model = ChatOllama(
    model=model_name,
    temperature=0.5,
    streaming=True,
    model_kwargs={
        "base_url": "http://localhost:11434",
        "num_thread": 4,  # Reduce CPU core usage
        "num_gpu": 1,  # If you have GPU, keep this low
        "top_k": 20,  # small = faster
    },
)
logger.info("ChatOllama model initialized successfully")

template = """
You are an exeprt in answering questions about a pizza restaurant.
if the question is not related to the reviews, politely inform the user that you can only answer questions related to the reviews.
if there is not enough information in the reviews, say "I don't know".
Here are some relevant reviews:
{reviews}

Here is the question to answer:.
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | chat_ollama_model

import time


def answer_question_stream(question: str):
    logger.info(f"Answering question: {question}")

    # Retrieve docs normally
    relevant_docs = retriever.invoke(question)
    logger.info(f"documents retrieved: {relevant_docs}")
    reviews_text = "\n".join(
        [
            f"- {doc.page_content} (Rating: {doc.metadata['rating']}, Date: {doc.metadata['date']})"
            for doc in relevant_docs
        ]
    )

    logger.info(f"Compiled reviews text: {reviews_text[:200]}...")  # first 200 chars

    logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

    final_input = {
        "reviews": reviews_text,
        "question": question,
    }

    logger.info("\n🔵 Streaming answer:\n")
    start_time = time.time()
    # STREAM OUTPUT IN REAL TIME
    for chunk in chain.stream(final_input):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    end_time = time.time() - start_time
    logger.info(f"\n\n⏱️ Answer generated in {end_time:.2f} seconds.")


if __name__ == "__main__":
    # sample_question = "how are the vegan options?"
    sample_question = input(
        "Enter another question about the restaurant (or type 'exit' to quit): "
    )
    while sample_question != "exit" or sample_question != "quit":
        if sample_question.strip() == "":
            sample_question = input("Please enter a valid question: ")
        answer_question_stream(sample_question)
        sample_question = input(
            "Enter another question about the restaurant (or type 'exit' to quit): "
        )
