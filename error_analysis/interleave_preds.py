# script to interleave predicted gloss with gold data
import argparse

from format_data import load_data_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file")
    parser.add_argument("pred_file")
    args = parser.parse_args()
    return args


def interleave(gold_file: str, pred_file: str):
    gold_data = load_data_file(gold_file)
    pred_data = load_data_file(pred_file)
    for gold_line, pred_line in zip(gold_data, pred_data):
        gold_line.set_pred(pred_line.glosses)
        print(gold_line.interleave_str())


def main():
    args = get_args()
    interleave(args.gold_file, args.pred_file)


if __name__ == main():
    main()
