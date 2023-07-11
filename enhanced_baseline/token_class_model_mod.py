import os
import random
from typing import Optional

import click
from custom_tokenizers import tokenizers
from data import (
    create_encoder,
    load_data_file,
    load_git_dictionary,
    ModelType,
    prepare_dataset,
    write_predictions,
)
from datasets import DatasetDict
from encoder import load_encoder, MultiVocabularyEncoder, special_chars
from eval import eval_morpheme_glosses, eval_word_glosses
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    optimization,
    RobertaConfig,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
)
import wandb

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_model(encoder: MultiVocabularyEncoder, sequence_length):
    print("Creating model...")
    config = RobertaConfig(
        vocab_size=250002,
        # vocab_size=encoder.vocab_size(),
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[2]) + len(special_chars),
    )
    model = RobertaForTokenClassification.from_pretrained(
        "xlm-roberta-base", config=config, ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(encoder.vocab_size())
    # print(model.config)
    return model.to(device)


def create_trainer(
    model: RobertaForTokenClassification,
    dataset: Optional[DatasetDict],
    encoder: MultiVocabularyEncoder,
    batch_size,
    lr,
    max_epochs,
    max_steps=-1,
    ckpt_path="training-checkpoints",
):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        # print(preds)
        decoded_preds = encoder.batch_decode(preds, from_vocabulary_index=2)
        # print(decoded_preds[0:1])

        # Decode (gold) labels
        # print(labels)
        labels = np.where(labels != -100, labels, encoder.PAD_ID)
        decoded_labels = encoder.batch_decode(labels, from_vocabulary_index=2)
        # print(decoded_labels[0:1])

        if encoder.segmented:
            return eval_morpheme_glosses(
                pred_morphemes=decoded_preds, gold_morphemes=decoded_labels
            )
        else:
            return eval_word_glosses(
                pred_words=decoded_preds, gold_words=decoded_labels
            )

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=2)

    optimizer = optimization.Adafactor(
        model.parameters(),
        lr=lr,
        scale_parameter=True if lr is None else False,
        relative_step=True if lr is None else False,
        warmup_init=True if lr is None else False,
    )
    lr_scheduler = optimization.AdafactorSchedule(optimizer)

    args = TrainingArguments(
        output_dir=ckpt_path,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        weight_decay=1e-2,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        max_steps=max_steps,
        # metric_for_best_model="bleu",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
    )

    trainer = Trainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["dev"] if dataset else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    return trainer


languages = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
    "ntu": "Natugu",
}


@click.command()
@click.argument("mode")
@click.option("--lang", help="Which language to train", type=str, required=True)
@click.option(
    "--track",
    help="[closed, open] whether to use morpheme segmentation",
    type=str,
    required=True,
)
@click.option("--model_path", help="Path to (pretrained) model", type=click.Path())
@click.option(
    "--data_path",
    help="The dataset to run predictions on. Only valid in predict mode.",
    type=click.Path(exists=True),
)
def main(
    mode: str,
    lang: str,
    track: str,
    model_path: str,
    data_path: str,
):
    if mode == "train":
        wandb.init(project="sigmorphon2023", entity="wav2gloss")

    MODEL_INPUT_LENGTH = 514

    is_open_track = track == "open"
    print("IS OPEN", is_open_track)

    track_num = "2" if is_open_track else "1"
    train_data = load_data_file(
        f"../data/{languages[lang]}/{lang}-train-track{track_num}-uncovered"
    )
    dev_data = load_data_file(
        f"../data/{languages[lang]}/{lang}-dev-track{track_num}-uncovered"
    )

    print("Preparing datasets...")

    tokenizer_test = tokenizers["morpheme" if is_open_track else "word_no_punc"]
    tokenizer_train = tokenizers["gloss" if is_open_track else "word_no_punc"]

    if mode == "train":
        pretrained_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        random.shuffle(train_data)

        encoder = create_encoder(
            train_data,
            tokenizer=tokenizer_train,
            threshold=1,
            model_type=ModelType.TOKEN_CLASS,
            split_morphemes=is_open_track,
            use_copy_token=True,
            pretrained_tokenizer=pretrained_tokenizer,
        )
        os.makedirs(model_path, exist_ok=True)
        encoder.save(os.path.join(model_path, "encoder_data.pkl"))
        dataset = DatasetDict()
        dataset["train"] = prepare_dataset(
            data=train_data,
            tokenizer=tokenizer_train,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        dataset["dev"] = prepare_dataset(
            data=dev_data,
            tokenizer=tokenizer_train,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH)
        trainer = create_trainer(
            model,
            dataset=dataset,
            encoder=encoder,
            batch_size=32,
            lr=None,
            max_epochs=40,
            max_steps=8000,
            ckpt_path=os.path.join(model_path, "checkpoints"),
        )

        print("Training...")
        trainer.train()
        print(f"Saving model to ./{model_path}")
        trainer.save_model(f"./{model_path}")
        print(f"Model saved at ./{model_path}")
    elif mode == "predict":
        encoder_path = os.path.join(model_path, "encoder_data.pkl")
        encoder = load_encoder(encoder_path)
        if not hasattr(encoder, "segmented"):
            encoder.segmented = is_open_track
        print("ENCODER SEGMENTING INPUT: ", encoder.segmented)
        predict_data = load_data_file(data_path)
        predict_data_train = prepare_dataset(
            data=predict_data,
            tokenizer=tokenizer_train,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        predict_data_pred = prepare_dataset(
            data=predict_data,
            tokenizer=tokenizer_test,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        model = RobertaForTokenClassification.from_pretrained(model_path)
        trainer = create_trainer(
            model, dataset=None, encoder=encoder, batch_size=16, lr=2e-5, max_epochs=50
        )
        preds = trainer.predict(test_dataset=predict_data_train).predictions
        if lang == "git":
            post_correct_dict = load_git_dictionary("git_dict.csv")
        else:
            post_correct_dict = None
        post_correct_dict = None

        write_predictions(
            data_path,
            lang=lang,
            preds=preds,
            pred_input_data=predict_data_pred,
            encoder=encoder,
            from_vocabulary_index=2,
            post_correct_dict=post_correct_dict,
        )


if __name__ == "__main__":
    main()
