import os
from evaluate import load

class TestResults():
    
    def __init__(self):
        self._prepare_for_eval()
        
    def get_correctness(self, df, code_col="func_code"):
         
        references = df["test"].tolist()
        predictions = [[c] for c in df[code_col]]
        _, details = self.code_eval.compute(references=references, 
                                            predictions=predictions, 
                                            k=[1], num_workers=3, timeout=5)


        return [bool(code) and details[i][0][1]["passed"]
                         for i, code in enumerate(df[code_col])]
            

    def _prepare_for_eval(self):
        """ Change environement variables to be ready for evaluation. 
        
        Author note:
        Careful: this is dangerous. In my experiments I did put that into a trusted
        environement, for use in other settings, one should make sure the whole
        experiments are conducted into a trusted secure env.

        """

        self.code_eval = load("code_eval")
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def humaneval_split(humaneval):
    test_string = humaneval[humaneval.find("assert"):]
    test_string = test_string.replace("assert", "").strip()
    test_string = test_string.split(" and ")
    inputs, outputs = zip(*[test_case.split("==") 
                       for test_case in test_string])
        
    return inputs, outputs