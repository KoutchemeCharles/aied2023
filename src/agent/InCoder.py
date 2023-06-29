import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList, StoppingCriteria
)
from sklearn.model_selection import ParameterGrid
from typing import List


# signals padding
PAD = "<pad>"
# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

class Incoder():

    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = self.load_tokenizer()
        assert "facebook" in self.config.path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode(self, input, output=""):
        prompt = _create_prompt(input, output)
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1762)
        return prompt, encoding 

    def batch_decode(self, model_outputs, prompts=None):
        completions = self.tokenizer.batch_decode(model_outputs, 
                                                  clean_up_tokenization_spaces=False) 
        prompts = prompts if prompts is not None else ["" * len(completions)]
        decoded = []
        for completion, prompt in zip(completions, prompts):
            while completion.startswith(PAD):
                completion = completion[len(PAD):]
            if completion.startswith(BOS):
                completion = completion[len(BOS):]
                if EOM not in completion:
                    completion += EOM
                completion = completion[:completion.index(EOM) + len(EOM)]
                completion = completion[:-len(EOM)]

            if prompt:
                completion = completion[len(prompt):]
                full_answer = prompt.replace(make_sentinel(1), "")
                full_answer = full_answer.replace(make_sentinel(0), completion)
                decoded.append((completion, full_answer))
            else:
                decoded.append(completion)
    
        return decoded
    
    def get_gen_param_space(self, num_return_sequences):
        """ Get the search space of generation hyperparameters. """

        # Exploring different kinds of generation
        # Note that beam_search decoding strategies work better
        # for neural translation models 
        
        default = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.6,
            "max_new_tokens": 256,
            "num_return_sequences": num_return_sequences
        }

        # multinomial sampling
        mns_grid = {
            "do_sample": [True],
            "top_p": [0.95, 0.90],
            "temperature": [0.2, 0.4, 0.6, 0.8], 
            "max_new_tokens": [256],
            "num_return_sequences": [num_return_sequences]
        }

        param_grid = [mns_grid]
        
        return list(ParameterGrid(param_grid)), default
    

    def load_model(self, path=None):
        path = path if path is not None else self.config.path
        return AutoModelForCausalLM.from_pretrained(path)
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.path)
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "left"
        return tokenizer
    
    def get_generate(self, model, generation_config, stop_words=[EOM, "\ndef"]):
        """ Return a function which given input_ids, obtain the
        paramaters. """
        
        stop_words_encoded = None
        if stop_words is not None:
            stop_words_encoded = [self.tokenizer.encode(word, 
                                                              add_special_tokens=False) 
                                  for word in stop_words]
            
        def generate(input_ids):
            max_input_length = input_ids.size(1)
            stopping_criteria = StoppingCriteriaList()
            max_lengths = [max_input_length for l in range(len(input_ids))]
            ssc = StopWordsStoppingCriteria(max_lengths, stop_words_encoded)
            stopping_criteria.append(ssc)

            if "cuda" in str(self.device): input_ids = input_ids.cuda() 
            with torch.no_grad():
                gen_outputs = model.generate(input_ids, 
                                             generation_config=generation_config, 
                                             stopping_criteria=stopping_criteria)
                return gen_outputs

        return generate
    


def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"
    
def _create_prompt(text, expected_output=""):
    """ 
    Create an inference or training prompt from the 
    text (and the optional expected_output)
    """
    
    parts = text.split("<infill>")
    prompt = ""
    # encode parts separated by sentinel
    for sentinel_ix, part in enumerate(parts):
        prompt += part
        prompt += make_sentinel(sentinel_ix)

    prompt += make_sentinel(0)
    if expected_output:
        prompt += expected_output
        prompt += EOM 

    return prompt

def encode_labels(input_ids, tokenizer):
    """ We want to consider the loss only for
    the non special tokens. 
    
    Predicates tells you the elements to ignore in the loss
    computed.
    
    """

    ignored_tokens = [PAD, make_sentinel(0), make_sentinel(1)]
    decoded = tokenizer.batch_decode(input_ids)
    labels = [-100 if d in ignored_tokens  else iid
              for iid, d in zip(input_ids, decoded)]
    return labels



class StopWordsStoppingCriteria(StoppingCriteria):
    def __init__(self, init_lengths: List[int], stop_words_encoded: List[List[int]]):
        super().__init__()
        self.init_lengths = init_lengths
        if stop_words_encoded is None:
            stop_words_encoded = []
        else:
            assert isinstance(stop_words_encoded[0], list)
        assert isinstance(stop_words_encoded, list)
        self.stop_words_encoded = stop_words_encoded

    def _contains_stop_words(self, tokens: List[int]):
        if not bool(self.stop_words_encoded):
            return False
        for start_ix in range(len(tokens)):
            for swe in self.stop_words_encoded:
                if tokens[start_ix:start_ix+len(swe)] == swe:
                    return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for init_length, i_tokens in zip(self.init_lengths, input_ids):
            if not self._contains_stop_words(i_tokens[init_length:].tolist()):
                return False
        return True
