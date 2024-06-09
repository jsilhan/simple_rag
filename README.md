# Simple RAG app

Simple API endpoint to query our knowledge base saved in files.

## Setup

* Create an `.env` file and add into it a record with your OpenAI API key, e.g. `OPENAI_API_KEY=sk-...`.
* Install all the dependencies with `poetry install`.
* Put the files you what to query into /data folder.
* Execute `poetry run create_vector_index` to load files from the directory into vector db.

## Run server

Start a server in a new terminal tab:
```bash
poetry run server
```

## Usage

To ask question to the knowledge base use:
```bash
curl -X POST -H 'Content-Type: application/json' "http://127.0.0.1:8000/ask" -d '{"message": "which countries I can sell to?"}'
```

### REST API

To connect the service with your own application via REST API see swagger doc: http://127.0.0.1:8000/docs#/default/ask_ask_post (the server must run).