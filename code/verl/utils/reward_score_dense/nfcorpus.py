import json
import logging
import os
import random
import re
import sys

from verl.utils.paths import DATA_DIR_ENV, data_path

logger = logging.getLogger(__name__)

sys.path.append('./')


from src.Dense.nfcorpus.search import PyseriniFaissSearcher
from src.Lucene.utils import ndcg_at_k

index_dir = str(data_path('local_index_search/nfcorpus/dense_index/faiss-flat.beir-v1.0.0-nfcorpus.bge-base-en-v1.5.20240107'))
model_path = "BAAI/bge-base-en-v1.5"

if not os.path.exists(index_dir):
    logger.warning('Faiss index not found for %s at %s (data dir: $%s, default <code>/data)',
                   'nfcorpus', index_dir, DATA_DIR_ENV)
    search_system = None
else:
    search_system = PyseriniFaissSearcher(index_dir, model_path)


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        logger.debug('Failed to locate model response header')
        return None, ""

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        logger.debug('No valid answer tags found')
        return None, processed_str
        

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        logger.debug('[Structure Validation]')
    validation_passed = True

    # processed_str = '<think> </think>' + processed_str
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            logger.debug('%s: count=%s, position=%s', tag_str, count, pos)
        
        if count != expected_count:
            if do_print:
                logger.debug('%s appears %s times (expected %s)', tag_str, count, expected_count)
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            logger.debug('Incorrect tag order: Expected <think>...</think><answer>...</answer>')
        validation_passed = False
    else:
        if do_print:
            logger.debug('Tag sequence validation passed')
    
    return validation_passed


def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                logger.debug('Empty JSON string')
            return False
        
        data = json.loads(json_str)
        
        # Required keys
        required_keys = {"query"}
        if not all(key in data for key in required_keys):
            if do_print:
                logger.debug('Missing required keys in JSON')
            return False

        return True
    except (json.JSONDecodeError, TypeError):  # TypeError: `in` on a JSON number / null
        if do_print:
            logger.debug('JSON decoding failed')
        return False

def retriver_items(query, top_k=3000, threads=16):
    """Retrieve items from the search system."""
    assert search_system is not None, "calculate_answer_score rejects a missing index first"
    results = search_system.search(query, top_k=top_k, threads=threads)
    return results
    
def calculate_answer_score(json_str, label, scores, top_k, do_print=False):
    """Calculate answer score based on final_prediction idx."""
    if search_system is None:
        raise RuntimeError(f"Search index for {__name__} was not loaded (see the import-time warning); "
                           "build it before computing rewards")
    if len(label) == 0:
        raise ValueError(f"{__name__} got an empty label (no relevant documents)")
    # Malformed model output scores -2.
    try:
        query = json.loads(json_str)['query']
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.debug('Error in evaluation')
        return -2
    
    # Executing the model's query must never crash training: any failure is logged and scored
    # -2, exactly as before. Misconfiguration was already rejected above.
    try:
        target = label
        results = retriver_items(query, top_k=top_k, threads=32)
        pred_results = [item[0] for item in results]
        answer_score = ndcg_at_k(pred_results, target, top_k, rel_scores=scores)
    except Exception:
        logger.exception("Search/scoring failed for model query %r; scoring -2", str(query)[:200])
        return -2

    return answer_score

def compute_score(solution_str, ground_truth, data_source, format_reward=0.1, answer_reward=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    label = [str(x) for x in ground_truth['target']]
    scores = ground_truth['score']
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    json_format_correct = check_json_format(answer_text, do_print)
    format_correct = response_format_correct and json_format_correct
    
    format_score = format_reward if format_correct else -2
    # if do_print:
    #     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    #     print(f"Format score: {format_score}")
    
    if do_print:
        logger.info('--------------------------------\nSolution string: %s\nTarget: %s |', solution_str, label)
    
    answer_score = 0
    if 'test' in data_source or 'val' in data_source:
        top_k = 10
    else:
        top_k = 100
    
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, scores, top_k, do_print)

    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score
    
    if do_print:
        logger.info('\n%s\n%s\n  Format: %s\n  Answer: %s\n  Total: %s\n%s', "-" * 80, " Final Score ".center(80, '-'),
                    format_score, answer_score, total_score, "=" * 80)

    return total_score


if __name__ == '__main__':
    solution_str = """<|im_start|>assistant:  <answer>{"query": "Microstructural development of human"}</answer>
"""
    ground_truth = {'target': ['4983'], 'score': [1]}
    scores = compute_score(solution_str, ground_truth, data_source='nfcorpus_dense')
    print(scores)