import os
from pathlib import Path
from sys import stderr

from openai import OpenAIError
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
)
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from simple_rag.settings import DB_COLLECTION_NAME, QDRANT_DB_PATH

MarkdownReader = download_loader("MarkdownReader")

load_dotenv()


def main():
    print("Storing vector index from files in /data")
    # for easier setup db is saved on a disk, on production we will use the server solution
    client = qdrant_client.QdrantClient(path=Path(os.getcwd()) / QDRANT_DB_PATH)
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    vector_store = QdrantVectorStore(client=client, collection_name=DB_COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    reader = SimpleDirectoryReader(
        input_dir="data",
        file_extractor={".md": MarkdownReader()},
    )
    docs = reader.load_data()
    try:
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, service_context=service_context)
    except OpenAIError as e:
        print(f"Can't save documents into db: {e}. Set the correct Open AI API key in .env file.", file=stderr)
        exit(1)
    index.storage_context.persist()


if __name__ == "__main__":
    main()
