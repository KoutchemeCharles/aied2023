""" Preparing an Huggingface dataset for execution with Refactory """

import os 
import time
from argparse import ArgumentParser
from dotmap import DotMap
from datasets import (
    load_dataset, Dataset,
    concatenate_datasets, Value
)
from src.utils.code import keep_unique_solutions
from src.utils.files import create_dir, read_config
from src.refactory.Refactory import Refactory


def main():
    args = parse_args()
    config = read_config(args.config)

    dataset = [
        get_buggy_submissions(config),
        get_reference_submissions(config)
    ]

    test_assignments = set(dataset[-1]["assignment_id"])

    corr_ds = get_correct_submissions(config)
    f = lambda ex: ex["assignment_id"] in test_assignments
    corr_ds = corr_ds.filter(f)
    if len(corr_ds.to_pandas()):
        # The tool does not need multiple versions of the same correct solution 
        corr_ds = keep_unique_solutions(corr_ds)
        dataset.append(corr_ds)

    dataset = concatenate_datasets(dataset)
    ref_config = DotMap({
        "save_path": config.save_dir, 
        "tool_path": config.tool_path
        })


    save_dir = os.path.join(config.save_dir, "refactory")
    create_dir(save_dir)

    # Saving the training dataset
    filename = f"{config.name}_dataset.csv"
    save_path = os.path.join(save_dir, filename)
    dataset.to_csv(save_path, index=False)

    # running and saving the results 
    results_dataset = Refactory(ref_config).run(dataset)
    filename = f"{config.name}_results.csv"
    save_path = os.path.join(save_dir, filename)
    results_dataset.to_csv(save_path, index=False)
    
   
def get_buggy_submissions(config):
    """ 
    Obtain the buggy student submissions to repair
    from a given dataset split. 
    """

    ds_name, split = config.buggy
    data = load_dataset("koutch/intro_prog",
                        f"{ds_name}_data")
    data_ds = data[split]
    data_ds = data_ds.filter(lambda ex: not ex["correct"])
    
    if config.remove_duplicates:
        data_ds = keep_unique_solutions(data_ds)
    if config.selected_assignments:
        f = lambda ex: ex["assignment_id"] in config.selected_assignments
        data_ds = data_ds.filter(f)

    return data_ds 


def get_reference_submissions(config):
    """ 
    We add the reference solutions 
    from the metadata split of the *test* dataset
    """

    ds_name, split = config.buggy
    metadata = load_dataset("koutch/intro_prog",
                            f"{ds_name}_metadata")
    metadata_ds = metadata[split]

    # We need to align the features of the metadata with the data 
    new_column = ["reference"] * len(metadata_ds)
    metadata_ds = metadata_ds.add_column("user", new_column)
    new_column = [-1] * len(metadata_ds)
    metadata_ds = metadata_ds.add_column("submission_id", new_column)
    new_column = [True] * len(metadata_ds)
    metadata_ds = metadata_ds.add_column("correct", new_column)
    metadata_ds = metadata_ds.rename_column("reference_solution", "func_code")
    new_features = metadata_ds.features.copy()
    new_features["submission_id"] = Value("int32")
    metadata_ds = metadata_ds.cast(new_features)

    return metadata_ds


def get_correct_submissions(config):
    """
    Fetch the correct solutions to search for repair
    from the training part (if there is)
    """

    data_ds = Dataset.from_dict({}) 
    if config.correction:
        ds_name, split = config.correction
        data = load_dataset("koutch/intro_prog",
                            f"{ds_name}_data")
        data_ds = data[split]
        data_ds = data_ds.filter(lambda ex: ex["correct"])

    return data_ds 


def parse_args():
    description = "Preparing a dataset for execution with Refactory"
    parser = ArgumentParser(description=description)
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    main()
