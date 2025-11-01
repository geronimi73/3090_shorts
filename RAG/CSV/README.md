## CSV-RAG
A minimal RAG implementation for querying CSV data using natural language. 

This script converts CSV files to SQLite and uses LLM tool calling to answer questions about your data—no complex frameworks required.

### Why?

Most RAG implementations rely on heavyweight frameworks like LangChain. This script demonstrates that you can build a functional CSV RAG system in ~200 lines of pure Python, making it perfect for:
* Learning how RAG and tool calling work under the hood
* Quickly prototyping data analysis workflows
* Understanding LLM agent patterns without framework abstractions

### How?

* Convert .csv to SQLite db
* Automatically generates column descriptions with sample values
* LLM call with single tool `run_sql`
* Two LLM options: Local Qwen or Mistral API
* Iterative LLM Calls until it comes up with an answer 

Dependencies: pandas, sqlite3, mistralai, transformers

### Quickstart

```bash
pip3 install -r requirements.txt

# Local `Qwen/Qwen3-4B-Instruct-2507`
python3 csv-rag.py --csv_file healthcare-dataset-stroke-data.csv --llm qwen --question "How many rows are there?"

# `mistral-small` via API
export MISTRAL_API_KEY=YOUR_KEY
python3 csv-rag.py --csv_file healthcare-dataset-stroke-data.csv --llm mistral-api --question "What's the average age?"
```


