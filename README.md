# Neural Automated Program Repair

Repository for the experiments conducted in the papers
[Training Language Models for Programming Feedback Using Automated Repair Tools](https://link.springer.com/chapter/10.1007/978-3-031-36272-9_79)
and [Automated Program Repair using Generative Models for Code Infilling](https://link.springer.com/chapter/10.1007/978-3-031-36272-9_74).

### Training Language Models for Programming Feedback Using Automated Repair Tools

In this paper, we show a simple strategy to instantiate a sequence-to-sequence model for
repairing student programs. The strategy consists of finetuning existing open models, 
such as those available on HuggingFace, using as ground truth the repairs found by
Automated Repair Tools. We use CodeT5 as the sequence-to-sequence model. 

### Automated Program Repair using Generative Models for Code Infilling

In this paper, we experiment with using Code Generative Models augmented with infilling
capabilities for the same task: repairing student programs. We conduct our experiments
with the InCoder model from Facebook. 

### Models and Dataset

The student datasets used to conduct our experiments, as well as the model
weights are available on [HuggingFace](https://huggingface.co/datasets/koutch/intro_prog).

## Installation

The requirements.txt file can be used with pip to get the packages needed to run the codes.

```
pip install -r requirements.txt
```

## Experiments

Before running the experiments, make sure to adapt the configurations files to
specify at least a saving directory, and, potentially, the path towards the location
of your installation of the [Refactory](https://github.com/githubhuyang/refactory) automated repair tool. 

Training and evaluating the sequence to sequence model on the dublin dataset,
and evaluating it's performance on three test datasets. 

```python
python run.py --experiment seq2seq --config config/seq2seq/dublin.json --train --test
python run.py --experiment seq2seq --config config/seq2seq/singapore.json --test
python run.py --experiment seq2seq --config config/seq2seq/newcaledonia.json --test
```

Training and evaluating the GMCI model on the dublin data

```python
python run.py --experiment gmci --config config/gmci/dublin.json
python run.py --experiment gmci --config config/gmci/singapore.json
python run.py --experiment seq2seq --config config/gmci/newcaledonia.json --test
```

The final results can be obtained by running the results.ipynb jupyter notebooks
