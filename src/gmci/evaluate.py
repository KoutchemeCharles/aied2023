import random
import numpy as np
from src.utils.code import does_compile
from src.utils.extract import get_predicted_function
from src.gmci.github_prompting import multiline_infilling_generator
from src.utils.distance import seq_dist
from transformers import GenerationConfig

def _evaluate(self, generate, dataset):
    
    def get_repairs(example):
        generator = multiline_infilling_generator(example)
        all_splits = [text for text, _, _ in generator]
        text = random.choice(all_splits)
        example["generations"] = self._inference(generate, text, example["func_name"])
        return example
    
    dataset = dataset.with_format("pytorch")
    eval_ds = dataset.map(get_repairs)
    pass_at_k, details = self.code_eval.compute(references=eval_ds["test"], 
                                                predictions=eval_ds["generations"], 
                                                k=self.config.heval_k, 
                                                num_workers=2, timeout=3) 
    results = {
        "experiment": "seq2seq",
        "eval_ds": eval_ds.to_dict(),
        "pass_at_k": pass_at_k,
        "details": details,
    }

    return results


def repair_code(self, example, generate):
    example["repair"] = ""

    all_combinations = list(multiline_infilling_generator(example))
    span_to_text = {span: text for (text, _, span) in all_combinations}
    repairs = []
    b, already_tried = 0, set()
    min_start, max_end = 1, len(example["func_code"].splitlines())
    
    while all_combinations and b < self.config.budget:

        # Get the next element of the list to try (LIFO order)
        (text, _, (start, end)) = all_combinations.pop()
        # Skip if we already tried that combination
        if (start, end) in already_tried:  
            continue
        # Skip if we found a repair outside of that span 
        elif start < min_start or end > max_end:
            continue
        
        already_tried.add((start, end))

        codes = self._inference(generate, text, example["func_name"])
        correctness = _is_correct(self.code_eval, codes, example["test"])
        repairs.extend([c for c, corr in zip(codes, correctness) if corr])

        # the next element we search will be half the ranges -> more efficient coverage
        middle = (start + end) // 2
        if middle != start and middle != end:
            all_combinations.append((span_to_text[(middle, end)], None, (middle, end)))
            all_combinations.append((span_to_text[(start, middle)], None, (start, middle)))

        if True in correctness:
            min_start, max_end = start, end
        else:
            # only increase budget if solution tried is not working 
            b = b + 1

    # Take the repair which minimize a given distance measure
    correction = ""
    if len(repairs):
        distances = [seq_dist(example["func_code"], c) for c in repairs]
        correction = repairs[np.argmin(distances)]

    example["repair"] = correction

    if correction != "":
        print("buggy code", example["func_code"])
        print("repair found", correction)

    return example
 

def _inference(self, generate, text, fname):
    prompt, encoding = self.agent.encode(text)
    model_outputs = generate(encoding.input_ids)
    prompts = [prompt] * len(model_outputs)
    outputs = self.agent.batch_decode(model_outputs, prompts)
    _, infilled = zip(*outputs)
    generations = []
    for infill in infilled:
        code = get_predicted_function(infill, fname)
        generations.append(code)

    return generations


def _is_correct(code_eval, codes, tests):
    """ Examine whether one constructed potential repair is correct. """
    
    references = [tests] * len(codes)
    predictions = [[code] for code in codes]
    _, details = code_eval.compute(references=references, 
                                        predictions=predictions, 
                                        k=[len(codes)], num_workers=1, timeout=3)

    return [does_compile(code) and details[i][0][1]["passed"]
             for i, code in enumerate(codes)]



def load_default_genconfig(self):
    _, default_gen_params = self.agent.get_gen_param_space(max(self.config.heval_k))
    generation_config = GenerationConfig(**default_gen_params)
    return generation_config
        