from src.agent.Seq2Seq import Seq2SeqAgent

class CodeT5(Seq2SeqAgent):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.tokenizer = self.load_tokenizer()
    
    def encode(self, example):
        if "description" in example:
            buggy_code = example["func_code"]
            description = add_comments_to_description(example["description"])
            text = buggy_code + "\n" + description + "\n"
            output = self.tokenizer(text)
        else:
            output = self.tokenizer(example["func_code"])
        
        if "repair" in example:
            output["labels"] = self.tokenizer(example["repair"])["input_ids"]

        return output 
    
    def _postprocess(self, decoded_code):
        if "def" in decoded_code:
            decoded_code = decoded_code.split("def")
            decoded_code = ["def" + c for c in decoded_code if c]
            decoded_code = decoded_code[0]
        return decoded_code


def add_comments_to_description(description):
    return "\n".join(["#" + l for l in description.splitlines()])