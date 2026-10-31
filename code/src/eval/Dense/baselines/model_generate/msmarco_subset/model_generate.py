import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.paths import data_path

logger = logging.getLogger(__name__)

CACHE_DIR = "/srv/local/data/linjc/hub"

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto',)
    model.eval()
    return tokenizer, model


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=8):
    df = pd.read_parquet(data_path)

    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()

    # if targets[i] is a array, then convert to list
    for i in range(len(targets)):
        # if is a np.ndarray, then convert to list
        if isinstance(targets[i], np.ndarray):
            targets[i] = targets[i].tolist()
    
    model.to(device)
    generated_texts = {}
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # try:
                output_ids = model.generate(**tokenized_inputs, max_new_tokens=1024, do_sample=False)
            # except:
                # continue
        
        for i, output in enumerate(output_ids):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            
            # extract from \nAssistant:
            generated_text = generated_text.split("\nAssistant:")[1]
            
            # findall <answer> </answer>
            answer_pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(answer_pattern, generated_text, re.DOTALL)  # Use re.DOTALL to match multiline content

            if len(matches) > 0:
                generated_text = matches[0]
                try:
                    # json.loads(generated_text)
                    generated_text = json.loads(generated_text)['query']
                except Exception:
                    logger.debug("answer in row %d is not JSON with a 'query' key; using the raw answer text",
                                 batch_start + i, exc_info=True)
                    generated_text = matches[0]

            idx = batch_start + i
            generated_texts[idx] = {
                "generated_text": generated_text,
                "target": str(targets[idx])
            }
        
        with open(os.path.join(save_dir, f"{model_name}.json"), "w") as f:
            json.dump(generated_texts, f, indent=4)
    
    with open(os.path.join(save_dir, f"{model_name}.json"), "w") as f:
        json.dump(generated_texts, f, indent=4)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_path", type=str, default=str(data_path("local_index_search/msmarco_science/dense/test.parquet")))
    parser.add_argument("--model_name", type=str, default="Qwen-inst-msmarco_science")
    parser.add_argument("--save_dir", type=str, default="results_dense")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size)

if __name__ == "__main__":
    main()
