from collections import defaultdict
from tokenize_rt import src_to_tokens, tokens_to_src, Token

def is_parsable(upload):
    """ Checks whether or not a string
    is parsable into python tokens. """
    
    try:
        src_to_tokens(upload)
        return True
    except:
        return False

def find_next_indent(tokens):
    """ Finds the first INDENT token. """
    
    for i, token_info in enumerate(tokens):
        if token_info.name == "INDENT":
            return i
            
    return len(tokens) - 1 if len(tokens) > 0 else 0 

def find_last_dedent(tokens):
    """ Finds the position of the last dedent token scoping
    the function within the tokens. """
    
    n_indent = 1
    for i, token_info in enumerate(tokens):
        
        if token_info.name == "INDENT":
            n_indent += 1
        elif token_info.name == "DEDENT":
            n_indent -= 1
            
        if n_indent == 0:
            return i # simply locate the first double point

    return None

def scope_function(start_index, tokens):
    """ Given a starting index and a list of tokens of a source code,
    find the index at which there is the end of the first function
    encountered. """
    
    start_search = find_next_indent(tokens[start_index:]) + start_index + 1
    end_search = find_last_dedent(tokens[start_search:]) 
    # if there are no dedent, I should just finnish at the last one 
    if end_search is not None:
        end_index = start_search + end_search
    else:
        end_index = start_search + len(tokens)

    return end_index


def find_functions(tokens):
    """ Analyze the sequences of tokens of a source code to
    and yields the list of tokens of the functions defined
    inside that source code. """
    
    for i, token_info in enumerate(tokens):
        if token_info.src == "def":
            start_index = i
            end_index = scope_function(start_index + 1, tokens)
            func_tokens = tokens[start_index: end_index]

            yield func_tokens
            
def parse_upload(upload):
    """ Parse the submitted student code into a list of 
    functions with their tokens. """
    
    try:
        token_infos = src_to_tokens(upload) # The tokens infos
    except (BaseException) as e:
        token_infos = []

    functions_tokens_info = list(find_functions(token_infos))
    
    informations = []
    for tokens_info in functions_tokens_info:
       
        informations.append({
            "tokens": tokens_info,
            "string": tokens_to_src(tokens_info), # this function does not guarantee correct reconstruction
            "name"  : get_function_name(tokens_info)
        })

    return informations

def get_function_name(tokens):
    """ Given a list of tokenize_rt.tokens of a function, 
    return the name of the function. """
    for t in tokens:
        if t.name == 'NAME' and t.src != "def":
            return t.src 
    return ""

def separate_functions(code):

    functions = defaultdict(list)
    information = parse_upload(code)
    for info in information:
        functions[info["name"]].append(info["string"])

    return functions


def get_predicted_function(completion, fname):
    functions = separate_functions(completion)
    if not functions or fname not in functions:
        return completion
    return functions[fname][0].strip() # taking the first one 