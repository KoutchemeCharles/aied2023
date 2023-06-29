from src.Experiment import Experiment
from src.utils.files import save_json

class Seq2Seq(Experiment):

    def __init__(self, config, test=False) -> None:
        super().__init__(config, "seq2seq", test)
        self.correctness_column = "repair_correctness"
        
    from .finetune import _get_trainer
    from .evaluate import _evaluate

    def train(self):
        print("Loading the dataset")
        dataset_dict = self.load_training_dataset()
        print("Encoding the dataset")
        dataset_dict = dataset_dict.map(self.agent.encode)
        model = self._train(dataset_dict)
        self.save_generation_config(model, dataset_dict)

        return self 
    
    def evaluate(self):
        test_dataset = self.load_test_dataset()
        
        model = self.load_trained_model()
        model = model.to(self.device)
        generation_config = self.load_best_genconfig()
        generation_config.bos_token_id = model.config.bos_token_id
        generation_config.decoder_start_token_id = model.config.decoder_start_token_id
        
        generate = self.agent.get_generate(model, generation_config)
        results = self._evaluate(generate, test_dataset)
        save_json(results, self.results_save_path)
        
        return self
    
