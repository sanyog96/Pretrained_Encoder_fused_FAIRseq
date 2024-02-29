"""
Tokenize raw sentences with the BERT tokenizer.
"""

import sys
import argparse
from transformers import BertTokenizer

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    for line in sys.stdin:
        tokens = tokenizer.tokenize(line.strip())
        print(" ".join(['[CLS]'] + tokens + ['[SEP]']))

# def main(args):
#     tokenizer = BertTokenizer.from_pretrained(args.bert_model)
#     with open(args.filename, "r") as fread:
#         sentences = fread.readlines()
#     for line in sentences:
#         tokens = tokenizer.tokenize(line.strip())
#         print(" ".join(['[CLS]'] + tokens + ['[SEP]']))

def cli_main():
    parser = argparse.ArgumentParser(description="Tokenize raw sentences with the BERT tokenizer")
    parser.add_argument("-m", "--model", type=str, metavar='DIR', dest="bert_model",
                        required=True, help="path to the BERT model")
    # parser.add_argument("-f", "--file", type=str, dest="filename",
    #                     required=True, help="filename")
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    cli_main()
