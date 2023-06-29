import os
import time
import pandas as pd 
from warnings import warn
from src.utils.TestResults import TestResults
from src.utils.code import clean_code


def collate_individual_csvs(dir_path):
    questions = os.listdir(dir_path)
    questions = [q for q in questions if q.startswith("question")]
    key_f = lambda q: int(q.split('_')[-1])
    questions = sorted(questions, key=key_f)

    dataframe = []
    for q in questions:
        q_path = os.path.join(dir_path, q, 'refactory_online.csv')
        if not os.path.exists(q_path):
            warn(f"Results for assignment {q} are not available")
            continue
        dataframe.append(pd.read_csv(q_path))
        
    dataframe = pd.concat(dataframe, axis=0, ignore_index=True)
    dataframe["submission_id"] = dataframe["File Name"].apply(extract_index).astype(int)
    dataframe = dataframe.set_index("submission_id")
    dataframe = dataframe.sort_index()
    
    return dataframe
    
def merge_results_with_source(source_dataframe, results_dataframe):
    # We only take the incorrect ones
    source_dataframe = source_dataframe[~source_dataframe.correct]
    source_dataframe = source_dataframe.set_index("submission_id")
    source_dataframe = source_dataframe.sort_index()
    dataframe = pd.concat([source_dataframe, results_dataframe], axis=1)
    # We want to keep the information about which submission is which 
    dataframe = dataframe.reset_index(drop=False)
    dataframe = dataframe.rename(columns={"Refactored Correct Code": "repair"})
    dataframe.loc[pd.isnull(dataframe.repair), "repair"] = ""
    dataframe["repair"] = dataframe["repair"].apply(clean_code)
    # renaming of some of the columns, and cleaning the repairs
    columns = list(source_dataframe.columns)
    columns.append("repair")
    columns.append("submission_id")
    dataframe = dataframe[columns]

    return dataframe

def reexecute_repairs(dataframe):
    correctness = TestResults().get_correctness(dataframe, code_col="repair")
    dataframe["repair_correctness"] = correctness
    
    return dataframe

def extract_index(file_name):
    return int(file_name.split("_")[-1][:-3])
