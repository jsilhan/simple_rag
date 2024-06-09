from openai import OpenAIError
import qdrant_client
import logging
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel

from simple_rag.settings import DB_COLLECTION_NAME, OPEN_AI_MODEL_NAME, QDRANT_DB_PATH

load_dotenv()

logger = logging.getLogger(__name__)

start_up_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_up_context["query_engine"] = get_query_engine()
    yield
    start_up_context.clear()


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    message: str


class Answer(BaseModel):
    message: str


def get_query_engine() -> BaseQueryEngine:
    client = qdrant_client.AsyncQdrantClient(path=QDRANT_DB_PATH)
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=OPEN_AI_MODEL_NAME, temperature=0), embed_model=embed_model
    )
    vector_store = QdrantVectorStore(collection_name=DB_COLLECTION_NAME, aclient=client)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    ).as_query_engine(verbose=True)


@app.post("/ask")  # this is POST to not be limitted by GET querystring max length
async def ask(question: Question) -> Answer:
    """
    Chat with the knowledge base.

    Send a question in the request body to get a response from the RAG.

    - **question**: The question you want to ask.

    Returns:
    - **answer**: The JSON response with an answer in the "message" attribute.
    """
    if not question.message:
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        response = await start_up_context["query_engine"].aquery(question.message)
    except (OpenAIError, ValueError) as e:
        logger.exception(
            f"Wrong db/OpenAI credentials setup, an error occured during /ask call: {e}",
        )
        raise HTTPException(
            status_code=503, detail="Server can't process the request now. We are working on the fix. Try again later."
        )
    return Answer(message=str(response))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
