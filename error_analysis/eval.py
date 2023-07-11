"""Contains the evaluation scripts for comparing predicted and gold IGT"""

import json
from typing import List

import click
from format_data import load_data_file


def eval_accuracy(
    segs: List[List[str]], pred: List[List[str]], gold: List[List[str]], vocab: dict
) -> dict:
    """Computes the average and overall accuracy, where predicted labels must be
    in the correct position in the list."""
    total_correct_predictions = 0
    total_correct_in_vocab_preds = 0
    total_correct_oov_preds = 0
    total_tokens = 0
    total_in_vocab_tokens = 0
    summed_accuracies = 0

    for entry_segs, entry_pred, entry_gold, i in zip(
        segs, pred, gold, range(len(gold))
    ):
        entry_correct_predictions = 0
        in_vocab_entry_correct_predictions = 0
        oov_entry_correct_predictions = 0

        for token_index in range(len(entry_gold)):
            # For each token, check if it matches
            if (
                token_index < len(entry_pred)
                and entry_pred[token_index] == entry_gold[token_index]
                and entry_pred[token_index] != "[UNK]"
            ):
                entry_correct_predictions += 1
                # print(entry_segs[token_index] in vocab.keys())
                if (
                    entry_segs[token_index] in vocab.keys()
                    and entry_gold[token_index] in vocab[entry_segs[token_index]]
                ):
                    total_in_vocab_tokens += 1
                    in_vocab_entry_correct_predictions += 1
                else:
                    oov_entry_correct_predictions += 1
            else:
                if (
                    entry_segs[token_index] in vocab.keys()
                    and entry_gold[token_index] in vocab[entry_segs[token_index]]
                ):
                    total_in_vocab_tokens += 1

        entry_accuracy = entry_correct_predictions / len(entry_gold)
        summed_accuracies += entry_accuracy

        total_correct_predictions += entry_correct_predictions
        total_correct_in_vocab_preds += in_vocab_entry_correct_predictions
        total_correct_oov_preds += oov_entry_correct_predictions

        total_tokens += len(entry_gold)
        total_oov_tokens = total_tokens - total_in_vocab_tokens

    total_entries = len(gold)
    average_accuracy = summed_accuracies / total_entries
    overall_accuracy = total_correct_predictions / total_tokens
    overall_in_vocab = total_correct_in_vocab_preds / (
        total_in_vocab_tokens + 0.0000001
    )
    overall_oov = total_correct_oov_preds / (total_oov_tokens + 0.0000001)
    return {
        "average_accuracy": average_accuracy,
        "accuracy": overall_accuracy,
        "in_vocab_accuracy": overall_in_vocab,
        "oov_accuracy": overall_oov,
    }


def eval_stems_grams(pred: List[List[str]], gold: List[List[str]]) -> dict:
    perf = {
        "stem": {"correct": 0, "pred": 0, "gold": 0},
        "gram": {"correct": 0, "pred": 0, "gold": 0},
    }

    for entry_pred, entry_gold in zip(pred, gold):
        for token_index in range(len(entry_gold)):
            # We can determine if a token is a stem or gram by checking if it is
            # all uppercase
            token_type = "gram" if entry_gold[token_index].isupper() else "stem"
            perf[token_type]["gold"] += 1

            if token_index < len(entry_pred):
                pred_token_type = (
                    "gram" if entry_pred[token_index].isupper() else "stem"
                )
                perf[pred_token_type]["pred"] += 1

                if entry_pred[token_index] == entry_gold[token_index]:
                    # Correct prediction
                    perf[token_type]["correct"] += 1

    stem_perf = {
        "prec": 0
        if perf["stem"]["pred"] == 0
        else perf["stem"]["correct"] / perf["stem"]["pred"],
        "rec": perf["stem"]["correct"] / perf["stem"]["gold"],
    }
    if (stem_perf["prec"] + stem_perf["rec"]) == 0:
        stem_perf["f1"] = 0
    else:
        stem_perf["f1"] = (
            2
            * (stem_perf["prec"] * stem_perf["rec"])
            / (stem_perf["prec"] + stem_perf["rec"])
        )

    gram_perf = {
        "prec": 0
        if perf["gram"]["pred"] == 0
        else perf["gram"]["correct"] / perf["gram"]["pred"],
        "rec": perf["gram"]["correct"] / perf["gram"]["gold"],
    }
    if (gram_perf["prec"] + gram_perf["rec"]) == 0:
        gram_perf["f1"] = 0
    else:
        gram_perf["f1"] = (
            2
            * (gram_perf["prec"] * gram_perf["rec"])
            / (gram_perf["prec"] + gram_perf["rec"])
        )
    return {"stem": stem_perf, "gram": gram_perf}


def eval_morpheme_glosses(
    seg_morphemes: List[List[str]],
    pred_morphemes: List[List[str]],
    gold_morphemes: List[List[str]],
    vocab: dict,
):
    """Evaluates the performance at the morpheme level"""
    morpheme_eval = eval_accuracy(seg_morphemes, pred_morphemes, gold_morphemes, vocab)
    class_eval = eval_stems_grams(pred_morphemes, gold_morphemes)
    # bleu = bleu_score(pred_morphemes, [[line] for line in gold_morphemes])
    return {"morpheme_level": morpheme_eval, "classes": class_eval}


def eval_word_glosses(
    pred_words: List[List[str]], gold_words: List[List[str]], vocab: dict
):
    """Evaluates the performance at the word level"""
    word_eval = eval_accuracy(pred_words, gold_words)
    # bleu = bleu_score(pred_words, [[line] for line in gold_words])
    return {"word_level": word_eval}


@click.command()
@click.option(
    "--pred",
    help="File containing predicted IGT",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--gold",
    help="File containing gold-standard IGT",
    type=click.Path(exists=True),
    required=True,
)
@click.option("--lang", help="Language code of file", type=str, required=True)
def evaluate_igt(pred: str, gold: str, lang: str):
    """Performs evaluation of a predicted IGT file"""

    with open(f"vocabs/word/{lang}.json") as word_json:
        word_dict = json.load(word_json)

    with open(f"vocabs/morpheme/{lang}.json") as morph_json:
        morph_dict = json.load(morph_json)

    pred = load_data_file(pred)
    gold = load_data_file(gold)

    seg_words = [line.seg_list() for line in gold]
    pred_words = [line.gloss_list() for line in pred]
    gold_words = [line.gloss_list() for line in gold]
    word_eval = eval_accuracy(seg_words, pred_words, gold_words, word_dict)

    seg_morphemes = [line.seg_list(segmented=True) for line in pred]
    pred_morphemes = [line.gloss_list(segmented=True) for line in pred]
    gold_morphemes = [line.gloss_list(segmented=True) for line in gold]

    all_eval = {
        "word_level": word_eval,
        **eval_morpheme_glosses(
            seg_morphemes=seg_morphemes,
            pred_morphemes=pred_morphemes,
            gold_morphemes=gold_morphemes,
            vocab=morph_dict,
        ),
    }
    print(json.dumps(all_eval, sort_keys=True, indent=4))


if __name__ == "__main__":
    evaluate_igt()
