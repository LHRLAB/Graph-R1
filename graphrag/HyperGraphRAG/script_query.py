import re
import json
import asyncio
from hypergraphrag import HyperGraphRAG, QueryParam
from tqdm import tqdm
import os
import argparse
os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()

def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)

    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()
    
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(error_file):
        os.remove(error_file)

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in tqdm(queries, desc="Processing queries", unit="query"):
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls", type=str, default="hypertension")
    parser.add_argument("--level", type=str, default="hard")
    args = parser.parse_args()
    cls = args.cls
    level = args.level
    WORKING_DIR = f"expr/{cls}"

    rag = HyperGraphRAG(working_dir=WORKING_DIR)
    query_param = QueryParam(mode="hybrid")

    with open(f"datasets/questions/{cls}/{cls}_{level}.json", "r") as f:
        data = json.load(f)
    queries = [item["query"] for item in data]
    
    if not os.path.exists(f"output/HyperGraphRAG/{cls}/{level}"):
        os.makedirs(f"output/HyperGraphRAG/{cls}/{level}")
    
    run_queries_and_save_to_json(
        queries, rag, query_param, f"output/HyperGraphRAG/{cls}/{level}/{cls}_{level}_result.json", f"output/HyperGraphRAG/{cls}/{level}/{cls}_{level}_errors.json"
    )
