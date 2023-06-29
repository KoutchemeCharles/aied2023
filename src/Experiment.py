import os
import torch
import numpy as np
from evaluate import load
from datasets import load_dataset, Dataset
from transformers import GenerationConfig

from src.agent.InCoder import Incoder
from src.agent.CodeT5 import CodeT5
from src.utils.files import create_dir
from src.utils.extract import separate_functions
from src.utils.code import (
    keep_unique_solutions, remove_outliers_with_mad
)

class Experiment(object):
    
    def __init__(self, config, name, test=False) -> None:
        self.config = config
        self.name = name
        self.test = test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._prepare_agent_for_eval()
        self.agent = self.load_model_agent()

        # make sure the save paths exist
        if self.test: self.name += "_test"
        self.save_dir = os.path.join(self.config.save_dir, self.name)
        self.results_save_dir = os.path.join(self.save_dir, "results")
        self.model_save_dir = os.path.join(self.save_dir, "model")
        create_dir(self.save_dir)
        create_dir(self.results_save_dir)
        create_dir(self.model_save_dir)
        filename = f"{self.config.name}_results.json"
        self.results_save_path = os.path.join(self.save_dir, "results", filename)
        
        
    def save_generation_config(self, model, dataset_dict):
        dataset = dataset_dict["test"] # concatenate_datasets(list(dataset_dict.values()))
        nrs = max(self.config.heval_k) if self.config.heval_k else 1
        search_space, gen_params = self.agent.get_gen_param_space(nrs)

        if self.config.training.search_best_genconfig:
            model = model.to(self.device)
            gen_params = self._search_best_gen_params(model, dataset, search_space)
            print("Best generation parameters: ", gen_params)
        
        # save the best generation hyperparameters
        # at the same position than the model 
        generation_config = GenerationConfig(**gen_params)
        generation_config.save_pretrained(self.model_save_dir)
        
        return generation_config
    

    def load_training_dataset(self):
        dataset = self._load_dataset(self.config.training_configuration)
        dataset = dataset.filter(lambda ex: ex[self.correctness_column])

        # Remove the solutions which use helper functions
        f = lambda ex: len(separate_functions(ex["func_code"])) == 1
        dataset = dataset.filter(f)

        # Remove outlier solutions in terms of length
        dataset = remove_outliers_with_mad(dataset, "assignment_id")
        
        # Keeping the unique solutions 
        # dataset = keep_unique_solutions(dataset)

        # We keep a subset of the submissions for validation purposes (dev)
        dataset_dict = dataset.train_test_split(0.10, seed=self.config.seed)
        
        if self.test:
            dataset_dict["train"] = dataset_dict["train"].select(range(10))
            dataset_dict["test"] = dataset_dict["test"].select(range(10))

        return dataset_dict
    
    def load_test_dataset(self):
        """ Load the dataset used for testing the model. """

        # only keep the solutions which were not correct and our selection
        # of assignments 

        dataset = self._load_dataset(self.config.test_configuration)
        dataset = dataset.filter(lambda example: not example["correct"])
        if self.config.selected_assignments:
            f = lambda ex: ex["assignment_id"] in self.config.selected_assignments
            dataset = dataset.filter(f)


        # Remove the solutions which have multiple files
        f = lambda ex: len(separate_functions(ex["func_code"])) == 1
        dataset = dataset.filter(f)

        # Keeping the unique solutions 
        dataset = keep_unique_solutions(dataset)

        # Remove outlier solutions in terms of length
        if "assignment_id" in dataset.features:
            dataset = remove_outliers_with_mad(dataset, "assignment_id")
        
        if self.test:
            dataset = dataset.select(range(10))
        
        return dataset

    def _load_dataset(self, path):
        if type(path) == list and len(path) > 1:
            # load the dataset from the hub
            configuration, split = path
            return load_dataset("koutch/intro_prog", configuration)[split]
        else:
            # load the dataset from the disk
            if path.endswith(".csv"):
                return Dataset.from_csv(path)
            elif path.endswith(".json"):
                return Dataset.from_json(path)
            else:
                raise ValueError(f"Cannot load dataset from {path}")

    def _train(self, dataset_dict):
        trainer = self._get_trainer(dataset_dict)
        args = trainer.args
        
        if self.config.training.wandb_hp_space: # Sweep for good hyperparameters
            print("Looking for best hyperparameters")
            best_trial = self._search_best_training_params(trainer)
            training_params = best_trial.hyperparameters
        elif self.config.training.hyperparameters: # Use the specified hyperaparameters
            training_params = self.config.training.hyperparameters
        else:
            # TODO: does not work apparently
            training_params = args.to_dict()
        # Else: use the default trainer hyperparameters
        
        # Update the training arguments with the set hyperparameters 
        training_params = {k: v for k, v in training_params.items() if hasattr(args, k)}
        print("Best training parameters", training_params)

        trainer = self._get_trainer(dataset_dict, **training_params)
        print("Retraining the model for a final time.")
        trainer.train()
        trainer.save_model(self.model_save_dir)
        print("Saving model to", self.model_save_dir)

        return trainer.model 
    
    def _search_best_gen_params(self, model, dataset, gen_param_space):
        """ Searches for the generation parameters which will
        produce the best results, and save them on the hub.

        """

        # When searching for the best generation hyperparameters
        # we can use the pass rate as a measure of success since at this
        # stage (end of training) we expect the model to be somewhat good
        # at generating repairs
        scores = []
        for gen_params in gen_param_space:
            generation_config = GenerationConfig(**gen_params)
            generate = self.agent.get_generate(model, generation_config)
            print("evaluating config", generation_config)
            res = self._evaluate(generate, dataset)["pass_at_k"]
            scores.append(res[f"pass@{gen_params['num_return_sequences']}"])
            if self.test:
                break

        return gen_param_space[np.argmax(scores)]
    
    def _search_best_training_params(self, trainer):
        n_trials = 1 if self.test else self.config.training.n_trials 
        best_trial = trainer.hyperparameter_search(
                backend="wandb",
                hp_space=lambda trial: self.config.training.wandb_hp_space.toDict(),
                n_trials=n_trials
        )

        return best_trial
    

    def load_trained_model(self):
        return self.agent.load_model(self.model_save_dir)

    def load_best_genconfig(self):
        return GenerationConfig.from_pretrained(self.model_save_dir)

    def load_model_agent(self):
        """ Load the agent class object interfacing the model evaluated. """

        if self.config.agent.name == "code_t5":
            self.agent = CodeT5(self.config.agent)
        elif self.config.agent.name == "incoder":
            self.agent = Incoder(self.config.agent)
        else:
            raise NotImplementedError("Model Not yet present")
            
        return self.agent 
    
    def _prepare_agent_for_eval(self):
        """ Change environement variables to be ready for evaluation. 
        
        Author note:
        Careful: this is dangerous. In my experiments I did put that into a trusted
        environement, for use in other settings, one should make sure the whole
        experiments are conducted into a trusted secure env.

        """

        self.code_eval = load("code_eval")
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

