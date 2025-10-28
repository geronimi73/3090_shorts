import pandas, sqlite3, json, re, argparse, os
from pathlib import Path

def pandas_to_sqlite(dataframe, db_fn):
    """convert a pandas dataframe to sqlite db"""

    with sqlite3.connect(db_fn) as conn:
        dataframe.to_sql(name = Path(csv_fn).stem, con = conn, if_exists = "replace")

def pandas_to_schema(dataframe):
    """returns LLM/human readable schema of a pandas dataframe"""
    schema_info = []
    for col in dataframe.columns:
        unique_vals = dataframe[col].dropna().unique()
        dtype_str = 'text' if dataframe[col].dtype == 'object' else str(dataframe[col].dtype)
        if len(unique_vals) <= 5: schema_info.append(f"- {col}: {dtype_str} (values: {', '.join(map(str, unique_vals))})")
        else: schema_info.append(f"- {col}: {dtype_str}")

    return '\n'.join(schema_info)

def run_sql(query, db_fn):
    """run given query against sqlite database; LLM tool"""
    print(f"Running SQL query:\n{query}")
    try:
        # Connect in read-only mode
        with sqlite3.connect(f"file:{db_fn}?mode=ro", uri=True) as conn:
            results = conn.execute(query).fetchall()
    except Exception as e:
        # Return error right back to LLM 
        return str(e)

    return str(results)

# tool definition
# alternative: generate with transformers.utils.get_json_schema()
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

def parse_tool_calls(content: str):
    """parse the tool calls in a qwen response."""
    tool_calls = []
    for m in re.finditer(r"<tool_call>\s(.+)?\s</tool_call>", content):
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print("json.JSONDecodeError", m)
            pass
    assert tool_calls
    return {"role": "assistant", "tool_calls": tool_calls}

def ask_agentQ(question, model, tokenizer, db_fn, max_iter=5):
    """run local QWEN in a loop for given number of max. iterations"""
    chat_history = [
        dict(role="system", content = system_prompt),
        dict(role="user",   content = question),
    ]
    
    for i in range(max_iter):
        print(f"* Iteration {i}")
    
        text = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            tools = [ run_sql_tooldef ],
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=16384)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if "<tool_call>" in output:
            assistantMsg = parse_tool_calls(output) 
        else:
            assistantMsg = dict(role="assistant", content=output)
        
        if not "content" in assistantMsg and i == max_iter - 1:
            print(f"No agent answer after {max_iter} iterations. Increase max_iter")            
        elif "content" in assistantMsg:
            # LLM has something to say, we're done
            print(f"\nAgent says: {assistantMsg['content']}")
            break
        else:
            assert "tool_calls" in assistantMsg
            chat_history.append(assistantMsg)
    
            # only one tool call, only one tool
            for tool_call in assistantMsg["tool_calls"]:
                assert tool_call["function"]["name"] ==  "run_sql"
                tool_response = run_sql(db_fn=db_fn, **tool_call["function"]["arguments"])
        
                # for debug
                print (f"LLM used tool {tool_call['function']['name']}")
                print (f"Tool output: {tool_response}")
            
                # update chat history
                chat_history.append(dict(role="tool", name=tool_call["function"]["name"], content=tool_response)) 

def ask_agentM(question, client, db_fn, max_iter=5):
    """run Mistral via API in a loop for given number of max. iterations"""

    chat_history = [
        dict(role="system", content = system_prompt),
        dict(role="user",   content = question),
    ]

    for i in range(max_iter):
        print(f"* Iteration {i}")
        
        inference = client.chat.complete(
            model = "mistral-small-latest",
            temperature = 0.3,
            messages = chat_history,
            tools = [ run_sql_tooldef ],
            tool_choice = "auto" 
        )
        assistantMsg = inference.choices[0].message
    
        if not assistantMsg.content and i == max_iter - 1:
            print(f"No agent answer after {max_iter} iterations. Increase max_iter")            
        elif assistantMsg.content:
            # LLM has something to say, we're done
            print(f"\nAgent says: {assistantMsg.content}")
            break
        else:
            assert assistantMsg.tool_calls
            chat_history.append(assistantMsg)

            # only one tool call, only one tool
            for tool_call in assistantMsg.tool_calls:
                assert tool_call.function.name ==  "run_sql"
                tool_response = run_sql(db_fn=db_fn, **json.loads(tool_call.function.arguments))
        
                # for debug
                print (f"LLM used tool {tool_call.function.name}")
                print (f"Tool output: {tool_response}")
            
                # update chat history
                chat_history.append(dict(role="tool", name=tool_call.function.name, tool_call_id=tool_call.id, content=tool_response))

# same for QWEN and Mistral
system_prompt_template = """You are a helpful assistant that answers questions about a stroke prediction dataset. 

You have access to a SQLite database with a table called [healthcare-dataset-stroke-data] (note the square brackets due to hyphens in the name).

The table has the following columns:
{schema}

Only answer questions related to the data. 
Your answers should be grounded in the data. No data, no answer.

Don't ever calculate anything yourself, use SQL instead.

Don't forget to wrap the table name in brackets. Example `SELECT * from [table]`."""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG with CSV.')
    parser.add_argument('--csv_file', type=str, default="healthcare-dataset-stroke-data.csv", help='Path to the CSV file', )
    parser.add_argument('--llm', type=str, default='qwen', choices=['mistral-api', 'qwen'], help='LLM to use: "mistral" or "qwen" (default: mistral)')
    parser.add_argument('--question', type=str, default='How many women are in the dataset (absolute and percentage)', help='Your question about the CSV file (required)')
    args = parser.parse_args()

    print("Parsed arguments:")
    print(json.dumps(vars(args), indent=2))

    csv_fn, llm, question = args.csv_file, args.llm, args.question
    assert os.path.exists(csv_fn), "Input file does not exist"

    # question = "What's the average BMI of the patients with and without hypertension ?"
    # question = "Is drinking red wine a risk factor for diabetes?"
    # question = "is hypertension a risk factor for stroke?"

    # CSV to SQLite
    df = pandas.read_csv(csv_fn)
    pandas_to_sqlite(df, db_fn := Path(csv_fn).stem + ".db")

    # Put schema in system prompt
    system_prompt = system_prompt_template.format(schema=pandas_to_schema(df))

    if llm == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "Qwen/Qwen3-4B-Instruct-2507"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")

        ask_agentQ(question, model, tokenizer, db_fn, max_iter=5)
        
    else:
        from mistralai import Mistral
        client = Mistral(api_key="xxx")

        ask_agentM(question, client, db_fn, max_iter=5)

