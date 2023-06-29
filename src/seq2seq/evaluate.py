def _evaluate(self, generate, dataset):
    
    def get_repairs(example):
        output_ids = generate(example["input_ids"])
        example["generations"] = self.agent.decode(output_ids)
        return example
    
    dataset = dataset.map(self.agent.encode)
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