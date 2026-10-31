import json
import os

from verl.utils.paths import data_path


def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Create JSON document with a clear structure
            doc = {
                "id": data["_id"],  # Unique identifier for search results
                "contents": data['title'] + '\n' + data['text'],  # Required field for Pyserini
            }
            
            docs.append(json.dumps(doc))

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

    print(f"✅ Converted JSONL saved to {output_file}")


ori_data_dir = str(data_path("raw_data/nfcorpus/corpus.jsonl"))
output_file = str(data_path("local_index_search/nfcorpus/jsonl_docs/pyserini_corpus.jsonl"))

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Example Usage
convert_jsonl_for_pyserini(ori_data_dir, output_file)
