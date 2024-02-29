"""
Tokenize raw sentences with the RoBERTa tokenizer.
"""

import sys
import argparse
from transformers import RobertaTokenizer

def main(args):
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
    for line in sys.stdin:
        tokens = tokenizer.tokenize(line.strip())
        print(" ".join(['<s>'] + tokens + ['</s>']))

def cli_main():
    parser = argparse.ArgumentParser(description="Tokenize raw sentences with the RoBERTa tokenizer")
    parser.add_argument("-m", "--model", type=str, metavar='DIR', dest="roberta_model",
                        required=True, help="path to the RoBERTa model")
    # parser.add_argument("-f", "--file", type=str, dest="filename",
    #                     required=True, help="filename")
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    cli_main()
