import ast, textwrap
import time
from warnings import warn
from collections import defaultdict

import numpy as np 
from astor import to_source 
from datasets import Dataset
from python_minifier import minify
from tokenize_rt import src_to_tokens, tokens_to_src, Token
from scipy.stats import median_abs_deviation

def keep_unique_solutions(ds, code_co="func_code", fname_col="func_name"):
    """ Remove duplicate solutions in terms of a metric. """

    df = ds.to_pandas()
    pairs = df[[code_co, fname_col]].to_numpy()
    df["normalized"] = [code_uniqueness(code, func_name) 
                        for code, func_name in pairs]
    
    def add_representative(sub_df):
        """ Add the representative of the codes having the same appraoch. """
        if not sub_df.empty:
            sub_df["representative"] = sub_df[code_co].value_counts().index[0]
        else:
            sub_df["representative"] = sub_df[code_co]
        return sub_df
        
    groups = df.groupby([fname_col, "normalized"], as_index=False)
    new_df = groups.apply(add_representative)

    # Now, select only one of the codes which match the representative 
    new_df = new_df[new_df.representative == new_df[code_co]]
    new_df = new_df.drop_duplicates("representative", ignore_index=True, keep='last')

    return Dataset.from_pandas(new_df) 

    
def clean_code(code, remove_docstring=True):
    """ Minify and clean a source code.
    
    Remove comments, initial docstrings, and empty blank lines. 
    Additionally, add a new docstring to the code.
    """

    code = minify(code, rename_locals=False, 
                  remove_literal_statements=remove_docstring)
    return to_source(ast.parse(code)).strip()
    

def does_compile(code):
    return get_ast(code) is not None

def get_ast(code):
    try:
        return ast.parse(code)
    except:
        return None 
    
def separate_functions(code):
    functions = defaultdict(list)
    code_ast = get_ast(code)
    if code_ast:
        for node in ast.walk(code_ast):
            if isinstance(node, ast.FunctionDef):
                functions[node.name].append(ast.get_source_segment(code, node))

    return functions

def get_function_name(code):
    functions = separate_functions(code)
    return list(functions.keys())[0] if functions else ""

# Execution

def get_code_executables(code):
    """ Get the defined in the functions
    
    Parameters
    ----------
    code: string
        Contains the definition of the function (as well as possible auxiliary functions definitions)
        we want to execute. 
    
    Returns
    -------
    d: Dict
        Functions and 
        
    """

    dictionary = {}
    string = textwrap.dedent(code)
    exec(string, dictionary, dictionary)
    
    return dictionary


def code_uniqueness(code, fname):
    """ Returns a normalized version of the code which could be
    used later to compare functions equivalence. """
    
    # Inspired by
    # https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
    
    if get_function_name(code) != fname:
        warn(f"Code {code} canot have a unique value")
        return None
    
    executables = get_code_executables(code)
    if fname not in executables:
        raise ValueError(f"Function {code} could not be obtained")
    func = executables[fname]
    if func is None:
        raise ValueError(f"Function {code} could not be obtained")
        
    variables = func.__code__.co_varnames
    new_var_name = {var: f"x_{i}" for i, var in enumerate(variables)}
    dumped = ast.dump(ast.parse(code))
    for var in variables:
        dumped = dumped.replace(f"'{var}'", f"'{new_var_name[var]}'")
    return dumped




def match_variables(source, destination, func_name):
    """ Matches variables from destination with variables from source. """
    
    # Take all variables from source (incorrect code)
    # find their counterparts in destination
    src_func = get_code_executables(source)[func_name]
    src_arguments = src_func.__code__.co_varnames[:src_func.__code__.co_argcount]
     
    # match each argument in source with each argument in destination
    dest_func = get_code_executables(destination)[func_name]
    dest_arguments = dest_func.__code__.co_varnames[:dest_func.__code__.co_argcount]
    
    new_var_names = src_arguments[:len(dest_arguments)]
    new_var_names = {d: s for s, d in zip(src_arguments, dest_arguments)}
    
    dest_tokens = src_to_tokens(destination)
    
    offset = 0
    new_dest_tokens = []
    for t in dest_tokens:
        src = new_var_names.get(t.src, t.src) if t.name == "NAME" else t.src 
        new_token = Token(name=t.name, src=src, utf8_byte_offset=offset)
        new_dest_tokens.append(new_token)
        offset += len(src)
    
    return tokens_to_src(new_dest_tokens)
    




def ast_to_passen_repre(sc_ast):
    """ Transforms a Python AST into the representation
    used for computing the tree edit distance used in 
    the python-edit-distance library 
    """
    adj_list = []
    n_list = []
    i = 0
    
    def dfs(node, i):
        node_name = str(node.__class__.__name__)
        adj_list.append([])
        n_list.append(node_name)
        node_adj_list = []
        for j, c in enumerate(ast.iter_child_nodes(node)):
            dfs(c, i + 1 + j)
            node_adj_list.append(i + 1 + j)
        adj_list[i] = node_adj_list
        
    dfs(sc_ast, i)
    
    return n_list, adj_list


def remove_outliers_with_mad(dataset, column, treshold=2.5):
    df = dataset.to_pandas()
    df = df[[c for c in df.columns if "__index_level_0__" not in c]] 
    df["n_lines"] = df["func_code"].apply(count_lines)
    groups = df.groupby(column, as_index=False, group_keys=False)
    f = lambda gdf: filter_with_mad(gdf, treshold)
    return Dataset.from_pandas(groups.apply(f))


def filter_with_mad(group_df, treshold):
    n_lines = group_df.n_lines
    mad = median_abs_deviation(n_lines)
    mask = np.ones(len(n_lines), dtype=bool)
    if mad:
        mask = ((n_lines - n_lines.median()) / mad) <= treshold
    
    return group_df[mask]


def count_lines(code):
    return len(code.splitlines())
