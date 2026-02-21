import pandas, sqlite3, json, argparse, os, re
import huggingface_hub
from dotenv import load_dotenv
from pathlib import Path
import difflib


"""
CSV-RAG: Retrieval-Augmented Generation for CSV data using SQL and LLMs.

This script converts CSV files to SQLite databases and uses LLMs (per HF API)
with tool calling to answer natural language questions about the data.

Usage Examples:
    python csv-rag.py --csv_file healthcare-dataset-stroke-data.csv --question "How many rows are there?"
    python csv-rag.py --csv_file healthcare-dataset-stroke-data.csv --question "What's the average age?"
    
    # With default settings:
    python csv-rag.py

"""

# tool definitions
run_sql_tooldef = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Runs a SQL query on the stroke dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute"
                }
            },
            "required": ["query"]
        }
    }
}

refine_system_prompt_tooldef = {
    "type": "function",
    "function": {
        "name": "refine_system_prompt",
        "description": "Refines and rewrites the ENTIRE system prompt. Use this to reorganize, consolidate, deduplicate, or restructure the prompt when it becomes messy or redundant. Your new prompt should preserve all useful information but present it clearly and concisely. This replaces the entire prompt, so include everything important.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The complete refined system prompt (replaces current prompt entirely). Pass None if you don't want to make any changes."
                }
            },
            "required": ["content"]
        }
    }
}

def run_sql(query, db_fn, **_):
    "Run given query against sqlite database; LLM tool"
    print(f"Running SQL query: {query}")
    try:
        with sqlite3.connect(f"file:{db_fn}?mode=ro", uri=True) as conn:        # Connect in read-only mode
            results = conn.execute(query).fetchall()
    except Exception as e:
        return "SQL Error: " + str(e)       # Return error right back to LLM, fix it!

    return str(results)

def refine_system_prompt(content, prompt_file):
    "Refine/rewrite the system prompt; LLM tool"
    if content is None or content.strip() == "":
        return "No changes to system prompt."

    with open(prompt_file, 'r') as f:
        old_prompt = f.read()

    with open(prompt_file, 'w') as f:
        f.write(content)

    # Show diff
    diff = ''.join(
        difflib.unified_diff(
            old_prompt.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile='before',
            tofile='after',
            lineterm='\n'
    ))
    if diff.strip():
        print(f"\n{'='*60}\nSystem Prompt Changes:\n{'='*30}")
        for line in diff.splitlines():
            col = (
                "[91m" if line.startswith('-') and not line.startswith('---') else
                "[92m" if line.startswith('+') and not line.startswith('+++') else
                "[93m" if line.startswith('@@') else ""
            )
            print(f"\033{col}{line}\033[0m")  # Red for deletions
    else:
        print("No changes detected in system prompt.")

    return f"Successfully refined system prompt."

def ask_agent(question, client, db_fn, prompt_file, max_iter=5):
    "Run model via API in a loop for given number of max. iterations; returns final answer string or None"

    # Read the current system prompt from file (may have been updated by previous runs)
    with open(prompt_file, 'r') as f:
        current_system_prompt = f.read()

    chat_history = [
        dict(role="system", content = current_system_prompt),
        dict(role="user",   content = question),
    ]
    prompt_updated = False

    for i in range(max_iter):
        print(f"* Iteration {i}")

        try:
            inference = client.chat.completions.create(
                model = "openai/gpt-oss-120b:groq",
                temperature = 1.0,
                top_p = 1.0,
                messages = chat_history,
                tools = [ run_sql_tooldef, refine_system_prompt_tooldef ],
                tool_choice = "auto"
            )
        except huggingface_hub.errors.BadRequestError as e:
            print(f"Invalid response. Retry ..")
            continue

        assistantMsg = inference.choices[0].message

        if not assistantMsg.content and i == max_iter - 1:
            print(f"No agent answer after {max_iter} iterations.")
            return None
        elif assistantMsg.content and not prompt_updated:
            # LLM answers but no system prompt refinement -> Enforce
            print(f"Enforcing refine_system_prompt")
            chat_history.append(dict(
                role="user",
                content="Before providing your final answer, you MUST call refine_system_prompt. Review your CURRENT system prompt carefully and rewrite it to include any new learnings from this query. For example, store learnings about the data structure, metadata. Do not store the data (or parts of it) in the prompt. Do not store any of the data . Keep it well-organized, concise, and free of redundancy."
            ))
        elif assistantMsg.content:
            # Final answer
            print(f"\nAgent says: {assistantMsg.content}")
            return assistantMsg.content
        elif assistantMsg.tool_calls:
            chat_history.append(assistantMsg)

            for tool_call in assistantMsg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name == "run_sql":
                    tool_response = run_sql(db_fn=db_fn, **func_args)
                elif func_name == "refine_system_prompt":
                    tool_response = refine_system_prompt(prompt_file=prompt_file, **func_args)
                    prompt_updated = True  
                else:
                    tool_response = f"Unknown tool: {func_name}"

                print (f"Tool use {func_name}. Output: {tool_response}")

                chat_history.append(
                    dict(role="tool", name=func_name, tool_call_id=tool_call.id, content=tool_response)
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG with CSV.')
    parser.add_argument('--csv_file', type=str, default="healthcare-dataset-stroke-data.csv", help='Path to the CSV file', )
    parser.add_argument('--question', type=str, default='How many women are in the dataset (absolute and percentage)?', help='Your question about the CSV file')
    parser.add_argument('--queries_json', type=str, help='Path to JSON file containing multiple queries to run sequentially')
    args = parser.parse_args()

    print("Parsed arguments:")
    print(json.dumps(vars(args), indent=2))

    load_dotenv()
    client = huggingface_hub.InferenceClient(token = os.getenv("HF_API_KEY"))

    # CSV to SQLite
    assert os.path.exists(args.csv_file), "CSV input file does not exist"
    df = pandas.read_csv(args.csv_file)
    with sqlite3.connect(db_fn := Path(args.csv_file).stem + ".db") as conn:
        df.to_sql(name=Path(args.csv_file).stem, con=conn, if_exists="replace")

    # Load/init system prompt
    prompt_file = "system_prompt.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            system_prompt = f.read()
        print(f"Loaded existing system prompt from {prompt_file}")
    else:
        with open(prompt_file, 'w') as f:
            f.write("You are a helpful assistant.")
        print(f"Created new system prompt file: {prompt_file}")

    if args.queries_json:
        print(f"\nRunning queries from {args.queries_json}\n")
        assert os.path.exists(args.queries_json), f"JSON file does not exist: {args.queries_json}"
        with open(args.queries_json, 'r') as f:
            queries_file = json.load(f)

        for i, query in enumerate(queries := queries_file["queries"], 1):
            question = query['question']

            print(f"{'='*30}\nQuery {i}/{len(queries)}\n{'='*30}")
            ask_agent(question, client, db_fn, prompt_file, max_iter=15)

        print(f"\nCompleted all {len(queries)} queries from JSON file.")
    elif args.question:
        print(f"\nRunning single query")
        ask_agent(args.question, client, db_fn, prompt_file, max_iter=15)



