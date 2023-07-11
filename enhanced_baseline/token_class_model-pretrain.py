import os
from typing import Optional

import click
from custom_tokenizers import tokenizers
from data import (
    create_encoder,
    load_data_file,
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
        # vocab_size=250002,
        vocab_size=encoder.vocab_size(),
        max_position_embeddings=sequence_length,
        pad_token_id=encoder.PAD_ID,
        num_labels=len(encoder.vocabularies[2]) + len(special_chars),
    )
    model = RobertaForTokenClassification.from_pretrained(
        "xlm-roberta-base", config=config, ignore_mismatched_sizes=True
    )
    # model.resize_token_embeddings(encoder.vocab_size())
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
    warmup=True,
):
    print("Creating trainer...")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predicted output
        print(preds)
        decoded_preds = encoder.batch_decode(preds, from_vocabulary_index=2)
        print(decoded_preds[0:1])

        # Decode (gold) labels
        print(labels)
        labels = np.where(labels != -100, labels, encoder.PAD_ID)
        decoded_labels = encoder.batch_decode(labels, from_vocabulary_index=2)
        print(decoded_labels[0:1])

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
        lr=None,
        scale_parameter=True,
        relative_step=True,
        warmup_init=warmup,
    )
    lr_scheduler = optimization.AdafactorSchedule(optimizer)

    args = TrainingArguments(
        output_dir="training-checkpoints",
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
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
    )

    trainer = Trainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler] if lr is None else (None, None),
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
@click.option(
    "--pretrained_path", help="Path to pretrained model", type=click.Path(exists=True)
)
@click.option(
    "--encoder_path", help="Path to pretrained encoder", type=click.Path(exists=True)
)
@click.option(
    "--data_path",
    help="The dataset to run predictions on. Only valid in predict mode.",
    type=click.Path(exists=True),
)
@click.option("--run_name", help="Name for wandb run", type=str)
@click.option(
    "--save_dir", help="Directory to save encoder and model weights", type=click.Path()
)
@click.option(
    "--use_translation", help="Use translation in training", default=True, type=bool
)
@click.option(
    "--exp_name", help="Name of experiment to append to output file", type=str
)
def main(
    mode: str,
    lang: str,
    track: str,
    pretrained_path: str,
    encoder_path: str,
    data_path: str,
    run_name: str,
    save_dir: str,
    use_translation: bool,
    exp_name: str,
):
    if mode == "pretrain":
        wandb.init(
            project="sigmorphon2023",
            entity="wav2gloss",
            group="pretraining",
            name=run_name,
        )
    if mode == "train" or mode == "train-no-pretrain":
        wandb.init(project="sigmorphon2023", entity="wav2gloss", name=run_name)

    MODEL_INPUT_LENGTH = 512

    is_open_track = track == "open"
    print("IS OPEN", is_open_track)

    track_num = "2" if is_open_track else "1"

    # TODO: need to change this later when we add different data augmentation
    pretrain_path = f"../data/{languages[lang]}/{lang}-train-artificial"
    train_path = f"../data/{languages[lang]}/{lang}-train-track{track_num}-uncovered"
    dev_path = f"../data/{languages[lang]}/{lang}-dev-track{track_num}-uncovered"

    train_data = load_data_file(train_path)
    dev_data = load_data_file(dev_path)

    print("Preparing datasets...")

    tokenizer = tokenizers["morpheme" if is_open_track else "word_no_punc"]
    AutoTokenizer.from_pretrained("xlm-roberta-base")

    if mode == "pretrain":
        pretrain_data = load_data_file(pretrain_path)
        encoder = create_encoder(
            pretrain_data + train_data,
            tokenizer=tokenizer,
            threshold=1,
            model_type=ModelType.TOKEN_CLASS,
            split_morphemes=is_open_track,
            use_copy_token=True,
        )
        os.makedirs(save_dir, exist_ok=True)
        encoder.save(os.path.join(save_dir, "encoder_data.pkl"))
        dataset = DatasetDict()
        dataset["train"] = prepare_dataset(
            data=pretrain_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        dataset["dev"] = prepare_dataset(
            data=dev_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
            # remove this line if translation is used in pretraining
            include_translation=False,
        )
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH)
        trainer = create_trainer(
            model,
            dataset=dataset,
            encoder=encoder,
            batch_size=16,
            lr=None,
            max_epochs=10,
        )

        print("Training...")
        trainer.train()
        print(f"Saving model to ./{save_dir}")
        trainer.save_model(f"./{save_dir}")
        print(f"Model saved at ./{save_dir}")

    elif mode == "train":
        encoder = load_encoder(encoder_path)
        dataset = DatasetDict()
        dataset["train"] = prepare_dataset(
            data=train_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
            include_translation=use_translation,
        )
        dataset["dev"] = prepare_dataset(
            data=dev_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
            include_translation=use_translation,
        )
        os.makedirs(save_dir, exist_ok=True)
        model = RobertaForTokenClassification.from_pretrained(pretrained_path)
        trainer = create_trainer(
            model,
            dataset=dataset,
            encoder=encoder,
            batch_size=16,
            lr=2e-5,
            max_epochs=5,
            warmup=False,
        )
        print("Training...")
        trainer.train()
        print(f"Saving model to ./{save_dir}")
        trainer.save_model(f"./{save_dir}")
        print(f"Model saved at ./{save_dir}")

    elif mode == "train-no-pretrain":
        encoder = create_encoder(
            train_data,
            tokenizer=tokenizer,
            threshold=1,
            model_type=ModelType.TOKEN_CLASS,
            split_morphemes=is_open_track,
            use_copy_token=True,
        )
        os.makedirs(save_dir, exist_ok=True)
        encoder.save(os.path.join(save_dir, "encoder_data.pkl"))
        dataset = DatasetDict()
        dataset["train"] = prepare_dataset(
            data=train_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        dataset["dev"] = prepare_dataset(
            data=dev_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
            # remove this line if translation is used in pretraining
            include_translation=False,
        )
        model = create_model(encoder=encoder, sequence_length=MODEL_INPUT_LENGTH)
        trainer = create_trainer(
            model,
            dataset=dataset,
            encoder=encoder,
            batch_size=8,
            lr=2e-5,
            max_epochs=40,
            warmup=False,
        )

        print("Training...")
        trainer.train()
        print(f"Saving model to ./{save_dir}")
        trainer.save_model(f"./{save_dir}")
        print(f"Model saved at ./{save_dir}")

    elif mode == "predict":
        encoder = load_encoder(encoder_path)
        if not hasattr(encoder, "segmented"):
            encoder.segmented = is_open_track
        print("ENCODER SEGMENTING INPUT: ", encoder.segmented)
        predict_data = load_data_file(data_path)
        predict_data = prepare_dataset(
            data=predict_data,
            tokenizer=tokenizer,
            encoder=encoder,
            model_input_length=MODEL_INPUT_LENGTH,
            model_type=ModelType.TOKEN_CLASS,
            device=device,
        )
        model = RobertaForTokenClassification.from_pretrained(pretrained_path)
        trainer = create_trainer(
            model, dataset=None, encoder=encoder, batch_size=8, lr=2e-5, max_epochs=50
        )
        preds = trainer.predict(test_dataset=predict_data).predictions
        write_predictions(
            data_path,
            lang=lang,
            preds=preds,
            pred_input_data=predict_data,
            encoder=encoder,
            from_vocabulary_index=2,
        )


if __name__ == "__main__":
    main()
