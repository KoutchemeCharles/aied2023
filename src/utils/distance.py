import ast
from edist.sed import (sed_string, standard_sed)
from edist.ted import standard_ted
from src.utils.code import ast_to_passen_repre
from tokenize_rt import src_to_tokens
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Classical distances

def str_dist(buggy, corrected):
    return sed_string(buggy, corrected)

def seq_dist(buggy, corrected):
    b_tokens = src_to_tokens(buggy)
    b_tokens = [t.src for t in b_tokens]
    c_tokens = src_to_tokens(corrected)
    c_tokens = [t.src for t in c_tokens]
    return standard_sed(b_tokens, c_tokens)

def ted_dist(buggy, corrected):
    x_nodes, x_adj = ast_to_passen_repre(ast.parse(buggy))
    y_nodes, y_adj = ast_to_passen_repre(ast.parse(corrected))
    return standard_ted(x_nodes, x_adj, y_nodes, y_adj)

# the normalized distances adapted versions
def str_norm_dist(buggy, corrected):
    return sed_string(buggy, corrected) / max(len(buggy), len(corrected))

def seq_norm_dist(buggy, corrected):
    b_tokens = src_to_tokens(buggy)
    b_tokens = [t.src for t in b_tokens]
    c_tokens = src_to_tokens(corrected)
    c_tokens = [t.src for t in c_tokens]
    return standard_sed(b_tokens, c_tokens) / max(len(b_tokens), len(c_tokens))

def ted_norm_dist(buggy, corrected):
    x_nodes, x_adj = ast_to_passen_repre(ast.parse(buggy))
    y_nodes, y_adj = ast_to_passen_repre(ast.parse(corrected))
    return standard_ted(x_nodes, x_adj, y_nodes, y_adj) / max(len(x_nodes), len(y_nodes))

# the relativee patch size distances
def str_rps_dist(buggy, corrected):
    return sed_string(buggy, corrected) / len(buggy)

def seq_rps_dist(buggy, corrected):
    b_tokens = src_to_tokens(buggy)
    b_tokens = [t.src for t in b_tokens]
    c_tokens = src_to_tokens(corrected)
    c_tokens = [t.src for t in c_tokens]
    return standard_sed(b_tokens, c_tokens) / len(b_tokens)

def ted_rps_dist(buggy, corrected):
    x_nodes, x_adj = ast_to_passen_repre(ast.parse(buggy))
    y_nodes, y_adj = ast_to_passen_repre(ast.parse(corrected))
    return standard_ted(x_nodes, x_adj, y_nodes, y_adj) / len(x_nodes)


# NLP adapted distances
def bleu_dist(buggy, corrected):
    b_tokens = src_to_tokens(buggy)
    b_tokens = [t.src for t in b_tokens]
    c_tokens = src_to_tokens(corrected)
    c_tokens = [t.src for t in c_tokens]
    return 1 - sentence_bleu([b_tokens], c_tokens)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    
def rouge1_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rouge1']
    return 1 - scores[-1]

def rouge2_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rouge2']
    return 1 - scores[-1]

def rougel_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rougeL']
    return 1 - scores[-1]

def rougelcsum_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rougeLsum']
    return 1 - scores[-1]
