from typing import Optional

import click
from data import get_collator, load_data_file, prepare_dataset, write_predictions
from datasets import concatenate_datasets, DatasetDict
import evaluate
import numpy as np
import torch
from tqdm import tqdm
import transformers
import wandb

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_model(pretrained):
    print("Creating model...")
    model = transformers.AutoModelForPreTraining.from_pretrained(pretrained)
    return model


def create_trainer(
    model,
    dataset: Optional[DatasetDict],
    tokenizer,
    batch_size,
    lr,
    max_epochs,
):
    print("Creating trainer...")
    metric = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.argmax(-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, word_order=2
        )
        result = {"chrf++": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    optimizer = transformers.optimization.Adafactor(
        model.parameters(),
        lr=None,
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
    )
    lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer)

    args = transformers.TrainingArguments(
        output_dir="training-checkpoints",
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=100,
        # bf16=True,
    )

    trainer = transformers.Trainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100
        ),
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["dev"] if dataset else None,
        compute_metrics=compute_metrics,
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

# lang_paths = languages
lang_paths = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi-romanized",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
    "ntu": "Natugu",
}

translations = {
    "arp": "English",
    "git": "English",
    "lez": "English",
    "nyb": "",
    "ddo": "English",
    "usp": "Spanish",
    "ntu": "English",
}


@click.command()
@click.argument("mode")
@click.option("--lang", help="Which language to train", type=str, required=True)
@click.option("--model_path", help="Path to model", type=click.Path())
def main(
    mode: str,
    lang: str,
    model_path: str,
):
    if mode == "train":
        wandb.init(project="sigmorphon2023", entity="wav2gloss")

    MODEL_INPUT_LENGTH = 512

    is_open_track = True
    print("IS OPEN", is_open_track)

    track_num = "2" if is_open_track else "1"
    pretrained_model = "google/byt5-base"

    if mode == "train":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/byt5-base", use_fast=False
        )

        if lang == "all":
            all_datasets = {"train": [], "dev": []}
            for lang in languages:
                print("Preparing datasets...")
                train_data = load_data_file(
                    f"../data/{lang_paths[lang]}/"
                    + f"{lang}-train-track{track_num}-uncovered"
                )
                dev_data = load_data_file(
                    f"../data/{lang_paths[lang]}/{lang}-dev-track{track_num}-uncovered"
                )
                collator = get_collator(
                    tokenizer=tokenizer,
                    src_lang=languages[lang],
                    transl_lang=translations[lang],
                    max_length=MODEL_INPUT_LENGTH,
                )
                train_data = prepare_dataset(data=train_data)
                all_datasets["train"].append(
                    train_data.map(
                        collator,
                        batched=True,
                        remove_columns=train_data.column_names,
                    )
                )
                dev_data = prepare_dataset(data=dev_data)
                all_datasets["dev"].append(
                    dev_data.map(
                        collator,
                        batched=True,
                        remove_columns=dev_data.column_names,
                    )
                )
            dataset = DatasetDict()
            dataset["train"] = concatenate_datasets(all_datasets["train"])
            all_datasets["dev"] = concatenate_datasets(all_datasets["dev"])
            rng = np.random.default_rng()
            dataset["dev"] = all_datasets["dev"].select(
                rng.choice(np.arange(all_datasets["dev"].num_rows), 400, replace=False)
            )
        else:
            dataset = DatasetDict()
            print("Preparing datasets...")
            train_data = load_data_file(
                f"../data/{languages[lang]}/{lang}-train-track{track_num}-uncovered"
            )
            dev_data = load_data_file(
                f"../data/{languages[lang]}/{lang}-dev-track{track_num}-uncovered"
            )

            collator = get_collator(
                tokenizer=tokenizer,
                src_lang=languages[lang],
                transl_lang=translations[lang],
                max_length=MODEL_INPUT_LENGTH,
            )
            dataset["train"] = prepare_dataset(data=train_data)
            dataset["train"] = dataset["train"].map(
                collator, batched=True, remove_columns=dataset["train"].column_names
            )
            dataset["dev"] = prepare_dataset(
                data=dev_data, max_lines=400, random_sample=True
            )
            dataset["dev"] = dataset["dev"].map(
                collator, batched=True, remove_columns=dataset["dev"].column_names
            )
        model = create_model(pretrained_model)
        trainer = create_trainer(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=4,
            lr=5e-5,
            max_epochs=20,
        )

        print("Training...")
        trainer.train()
        print(f"Saving model to {model_path}")
        trainer.save_model(model_path)
        print(f"Model saved at {model_path}")
    elif mode == "predict":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/byt5-base", use_fast=False
        )
        if lang == "all":
            langs = list(languages.keys())
        else:
            langs = [lang]
        for lang in langs:
            data_path = (
                f"../data/{lang_paths[lang]}/{lang}-test-track{track_num}-covered"
            )
            predict_data = load_data_file(data_path)
            predict_dataset = prepare_dataset(data=predict_data)
            collator = get_collator(
                tokenizer=tokenizer,
                src_lang=languages[lang],
                transl_lang=translations[lang],
                max_length=MODEL_INPUT_LENGTH,
            )
            predict_dataset = predict_dataset.map(
                collator, batched=True, remove_columns=predict_dataset.column_names
            )
            model = transformers.AutoModelForPreTraining.from_pretrained(model_path).to(
                device
            )

            preds = []
            for ex in tqdm(predict_dataset["input_ids"]):
                preds.append(
                    tokenizer.decode(
                        model.generate(
                            torch.tensor([ex]).to(device), max_length=MODEL_INPUT_LENGTH
                        )[0],
                        skip_special_tokens=True,
                    )
                )

            write_predictions(
                data_path,
                lang=lang,
                decoded_preds=preds,
            )


if __name__ == "__main__":
    main()
