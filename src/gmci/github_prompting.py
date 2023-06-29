def multiline_infilling_generator(example):
    """ 
    Return a generator of all contiguous prompts replacing
    each individual line in a code.
    """
    
    code = example['func_code']
    description = example['description']
    func_name = example['func_name']
    test = example['test']

    lines, indentation = get_lines(code)
    prefix = '<| file ext=.py |>'
    suffix = create_test_case_suffix(test, indentation)
    suffix += f"\n<|/ file filename={func_name}.py source=github dstars=0>"
    

    for length in range(1, len(lines)):
        start, end = 1, 1 + length
        while end <= len(lines):
            # merging the codes without the descriptions
            to_merge = lines[:start] + ["<infill>"] + lines[end:]
            new_code = f"\n{indentation}".join(to_merge)
            # adding the description as a docstring
            new_code = add_docstring(new_code, description)
            # adding the prefix, suffix
            prompt = prefix + "\n" + new_code + "\n" + suffix
            output = f"\n{indentation}".join(lines[start: end])

            yield prompt, output, (start, end)

            start, end = start + 1, end + 1


def get_lines(code):
    """ 
    Obtain the different lines in the code and the
    identation used. 
    """
    lines = code.splitlines()
    n_indents = len(lines[1]) - len(lines[1].lstrip())
    indentation = lines[1][:n_indents] 

    return lines, indentation
  
def create_test_case_suffix(tests, indentation):
    imports, tests_cases = tests.split("assert")
    tests_cases = tests_cases.strip().split("and")
    tests_cases = [f"{indentation}assert {t.strip()}" for t in tests_cases]
    
    suffix = """\nif __name__ == "__main__":\n"""
    if imports: suffix += f"{indentation}{imports.strip()}\n"
    suffix += f"\n".join(tests_cases)
    return suffix


def add_docstring(code, docstring):
    """ Add the given docstring to the code. """
    
    lines = code.split("\n")
    n_indents = len(lines[1]) - len(lines[1].lstrip())
    indentation = lines[1][:n_indents]
    lines.insert(1, add_indentation_to_docstring(docstring, indentation))

    return "\n".join(lines)


def add_indentation_to_docstring(docstring, indentation):
    """ Format the docstring such that the identation matches
    the code indentation. 
    """
        
    lines = docstring.strip().splitlines()
    lines = [l.strip() for l in lines]

    # First line is the summary
    summary = lines[0].strip()
    if len(lines) == 1:
        return f'{indentation}"""{summary}"""'
    
    # Then there is the rest
    lines = lines[:1] + [f"{indentation}{line}" for line in lines[1:]]
    docstring = "\n".join(lines)
    docstring = f'{indentation}"""{docstring}\n{indentation}"""'

    return docstring

