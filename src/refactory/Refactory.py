import os 
import tempfile
from subprocess import call
from datasets import Dataset
from src.refactory.preparation import create_save_dir
from src.refactory.analysis import (
    collate_individual_csvs,
    merge_results_with_source, reexecute_repairs
) 

class Refactory():

    def __init__(self, config) -> None:
        self.config = config 

    def run(self, dataset):
    
        with tempfile.TemporaryDirectory() as tmpdirname:
            create_save_dir(dataset, tmpdirname)
            run_tool(self.config.tool_path, tmpdirname)
            results_dataframe = collate_individual_csvs(tmpdirname)

        source_dataframe = dataset.to_pandas()    
        dataframe = merge_results_with_source(source_dataframe, 
                                              results_dataframe)
        dataframe = reexecute_repairs(dataframe)
        dataset = Dataset.from_pandas(dataframe, preserve_index=False)

        return dataset 
    
def run_tool(tool_path, dir_path):
    questions = os.listdir(dir_path)
    args = [
        "python3", f"{tool_path}/run.py",
        f"-d", dir_path, "-q", *questions,
        f"-s", "100", "-o", "-m", "-b", "-c"]
    call(args)
