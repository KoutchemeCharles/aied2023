import time
import torch 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.model_selection import ParameterGrid

class Seq2SeqAgent():

    def __init__(self, config) -> None:
        self.config = config 
        self.load_model()
        self.load_tokenizer()
    
    def encode(self, code):
        return self.tokenizer(self._preprocess(code))["input_ids"]

    def decode(self, output_ids):
        """ Obtain the decoded code of x model generations """
        output_codes = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print("output codes", output_codes)
        return list(map(self._postprocess, output_codes))

    def get_gen_param_space(self, num_return_sequences):
        """ Get the search space of generation hyperparameters. """

        # Exploring different kinds of generation
        # Note that beam_search decoding strategies work better
        # for neural translation models 
        
        default = {
            "num_beams": num_return_sequences,
            "do_sample": False,
            "max_new_tokens": 256,
            "length_penalty": 1.0,
            "num_return_sequences": num_return_sequences
        }

        # Beam Search
        bs_grid = {
            "num_beams": [num_return_sequences],
            "do_sample": [False],
            "max_new_tokens": [256],
            "length_penalty": [-0.5, 0.0, 1.0],
            "num_return_sequences": [num_return_sequences]
        }

        # Diverse Beam Search 
        dbs_grid = {
            "num_beams": [num_return_sequences],
            "num_beam_groups": [num_return_sequences],
            "max_new_tokens": [256],
            "length_penalty": [-0.5, 0.0, 1.0],
            "num_return_sequences": [num_return_sequences]
        }

        # multinomial sampling
        mns_grid = {
            "do_sample": [True],
            "top_p": [0.95],
            "temperature": [0.8], 
            "max_new_tokens": [256],
            "num_return_sequences": [num_return_sequences]
        }

        param_grid = [bs_grid, dbs_grid, mns_grid]
        
        return list(ParameterGrid(param_grid)), default

    def _preprocess(self, code):
        return code 

    def _postprocess(self, decoded_code):
        return decoded_code

    def load_model(self, path=None):
        path = path if path is not None else self.config.path
        return AutoModelForSeq2SeqLM.from_pretrained(path)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.path)
    
    def get_generate(self, model, generation_config):
        """ Return a function which given input_ids, obtain the
        paramaters. """
        
        def generate(input_ids):
            input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)
            with torch.no_grad():
                gen_outputs = model.generate(input_ids, generation_config=generation_config)
                return gen_outputs

        return generate