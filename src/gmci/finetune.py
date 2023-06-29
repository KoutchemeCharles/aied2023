import random
from transformers import (
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from src.agent.InCoder import _create_prompt, encode_labels
from src.gmci.github_prompting import multiline_infilling_generator


def _prepare_dataset_for_training(self, dataset_dict, tokenizer):

    # TODO: a more efficient way would be to change the sample at every epoch
    def sample_prompts(batch):
        example = {k: v[0] for k, v in batch.items()}
        generator = multiline_infilling_generator(example)
        all_splits = [(text, output) for text, output, _ in generator]
        samples = random.sample(all_splits, min(3, len(all_splits)))
        texts = [_create_prompt(text, output) for text, output in samples]
        """if self.test:
            print(texts[0])
            time.sleep(5)"""
        outputs = tokenizer(texts, padding='max_length', max_length=128, truncation=True)
        f = lambda iid: encode_labels(iid, tokenizer)
        outputs["labels"] = list(map(f, outputs.input_ids))
        return outputs
    
    dd = dataset_dict.map(sample_prompts, batched=True, batch_size=1, 
                          remove_columns=dataset_dict["train"].features)
    
    dd = dd.with_format("torch")
    return dd 

def _get_trainer(self, dataset_dict, **kwargs):
    """ We finetune the pre-trained model for a few epochs
    on our dataset of solutions. """

    tokenizer = self.agent.load_tokenizer()
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    args = TrainingArguments(
        self.model_save_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        save_strategy="no",
        # fp16=True, # when training InCoder, they say it's not a good idea to use half-precision
        seed=self.config.seed,
        report_to="wandb",
        **kwargs
    )

    trainer = Trainer(
        model=None,
        model_init=lambda trial: self.agent.load_model(),
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
    )
    
    return trainer 
