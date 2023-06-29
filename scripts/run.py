import gc 
import random, numpy, torch
from datasets import disable_caching
from argparse import ArgumentParser
from src.gmci.GMCI import GMCI
from src.seq2seq.Seq2Seq import Seq2Seq as SEQ2SEQ
from src.utils.files import read_config

def parse_args():
    parser = ArgumentParser(description="Running experiments of paper ...")
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument("--train",
                        help="Train a model on specified dataset stored on disk.",
                        action="store_true")
    parser.add_argument("--test",
                        help="Test a model on a specified configuration of a dataset.",
                        action="store_true")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")
    parser.add_argument('--experiment', choices=["gmci", "seq2seq"],
                        help="Which experiment to run.")

    return parser.parse_args()

def set_globals(config):
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    numpy.random.seed(config.seed)
    disable_caching()


def main():
    args = parse_args()
    config = read_config(args.config)
    set_globals(config)

    print("Configuration being run", config)
    
    experiment = globals()[args.experiment.upper()](config, args.test_run)
    if args.train:
        experiment.train()
    
    if args.test:
        gc.collect()
        torch.cuda.empty_cache()
        experiment.evaluate()
    

if __name__ == "__main__":
    main()
