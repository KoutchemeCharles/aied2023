from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def _get_trainer(self, dataset_dict, **kwargs):
    """ We finetune the pre-trained model for a few epochs
    on our dataset of solutions. """

    tokenizer = self.agent.load_tokenizer()
    model = self.agent.load_model()
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    args = Seq2SeqTrainingArguments(
        self.model_save_dir,
        overwrite_output_dir=True,
        evaluation_strategy = "epoch",
        save_strategy="no",
        fp16=True,
        seed=self.config.seed,
        report_to="wandb",
        **kwargs
    )

    trainer = Seq2SeqTrainer(
        model=None,
        model_init=lambda trial: self.agent.load_model(),
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
    )
    
    return trainer 
