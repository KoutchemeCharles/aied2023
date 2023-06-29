from src.Experiment import Experiment

class GMCI(Experiment):

    def __init__(self, config, test=False) -> None:
        super().__init__(config, "gmci", test)
        self.correctness_column = "correct"

    from .evaluate import (repair_code, _evaluate, _inference, load_default_genconfig)
    from .finetune import (_get_trainer,
                           _prepare_dataset_for_training)

    def train(self):
        """ 
        """

        dataset_dict = self.load_training_dataset()
        self.tokenizer = self.agent.load_tokenizer()
        # New: training the model 
        train_dd = self._prepare_dataset_for_training(dataset_dict, self.tokenizer)
        model = self._train(train_dd)
        # evaluate
        if self.device == "cuda": model = model.cuda()

        selection = list(range(min([len(dataset_dict["test"]), 100])))
        dataset_dict["test"] = dataset_dict["test"].shuffle().select(selection)
        self.save_generation_config(model, dataset_dict)

        return self 


    def evaluate(self):
        dataset = self.load_test_dataset()
        self.model = self.agent.load_model()
        # uncomment to load the trained version of the model 
        # self.model = self.load_trained_model()
        # Using half precision for evaluation 
        # self.model = self.model.half()
        if self.device == "cuda": self.model = self.model.cuda()
        self.tokenizer = self.agent.load_tokenizer()
        
        # load generation configuration 
        # uncomment to load default one 
        # best_gen_config = self.load_best_genconfig()
        best_gen_config = self.load_default_genconfig()
        print(self.model, best_gen_config, self.tokenizer)
        generate = self.agent.get_generate(self.model, best_gen_config)
        fn_kwargs = {"generate": generate}
        dataset = dataset.map(self.repair_code, fn_kwargs=fn_kwargs)
        dataset.to_json(self.results_save_path)
 