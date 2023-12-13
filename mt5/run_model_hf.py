import csv
import os

import datasets
import evaluate
import fire
import numpy as np
import torch
import transformers
from tqdm import tqdm

import wandb

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


def create_model(pretrained):
    print("Creating model...")
    model = transformers.AutoModelForPreTraining.from_pretrained(pretrained)
    return model


def _reassemble(sent):
    return " ".join(sent.strip().split())


def get_collator(tokenizer, max_length: int, use_translations=True):
    prompt1 = "Genrate interlinear gloss from {src_lang}: "
    prompt2 = ", with its translation: "

    def collate_fn(batch):
        nonlocal tokenizer, prompt1, prompt2, max_length
        inputs = [
            prompt1.format(src_lang=lang) + _reassemble(ex)
            for lang, ex in zip(batch["language"], batch["transcription"])
        ]
        if "translation" in batch and use_translations:
            for i, ex in enumerate(batch["translation"]):
                inputs[i] = inputs[i] + prompt2 + _reassemble(ex)
        inputs = [t + "\nAnswer: " for t in inputs]

        if "gloss" in batch:
            targets = [" ".join(ex.split()) for ex in batch["gloss"]]
        else:
            targets = None

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        return model_inputs

    return collate_fn


def create_trainer(
    model,
    dataset,
    tokenizer,
    batch_size,
    lr,
    max_epochs,
    ckpt_dir="training-checkpoints",
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

    args = transformers.Seq2SeqTrainingArguments(
        output_dir=ckpt_dir,
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        # gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=max_epochs,
        # load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=50,
        tf32=True,
        fp16_full_eval=True,
        predict_with_generate=True,
    )

    trainer = transformers.Seq2SeqTrainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100
        ),
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset["validation"] if dataset else None,
        compute_metrics=compute_metrics,
    )
    return trainer


def main(
    mode: str,
    dataset: str,
    model_path: str,
    use_translations: bool = True,
    preds_file: str = None,
):
    if mode == "train":
        wandb.init(project="fieldwork", entity="wav2gloss")

    MODEL_INPUT_LENGTH = 1024

    pretrained_model = "./byt5_odin"

    print(f"Use translations: {use_translations}")

    if mode == "train":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/byt5-base", use_fast=False
        )

        dataset = datasets.load_dataset(dataset, "all")
        collator = get_collator(
            tokenizer, max_length=MODEL_INPUT_LENGTH, use_translations=use_translations
        )
        dataset["train"] = (
            dataset["train"].filter(lambda x: x["discard"] == False).shuffle()
        )
        dataset["train"] = dataset["train"].map(
            collator, batched=True, remove_columns=dataset["train"].column_names
        )
        dataset["validation"] = (
            dataset["validation"].filter(lambda x: x["discard"] == False).shuffle()
        )
        dataset["validation"] = dataset["validation"].map(
            collator, batched=True, remove_columns=dataset["validation"].column_names
        )

        # print(f"Using a subset of odin with {dataset['train'].num_rows} rows")

        model = create_model(pretrained_model)
        trainer = create_trainer(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=2,
            lr=5e-5,
            max_epochs=10,
            ckpt_dir=os.path.join(model_path, "training-checkpoints"),
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
        collator = get_collator(
            tokenizer, max_length=MODEL_INPUT_LENGTH, use_translations=use_translations
        )
        print("Loading dataset...")
        dataset_test = datasets.load_dataset(dataset, "all", split="test")
        ids = dataset_test["id"]
        print("Generating predictions...")
        predict_dataset = dataset_test.map(
            collator, batched=True, remove_columns=dataset_test.column_names
        )
        model = transformers.AutoModelForPreTraining.from_pretrained(model_path).to(
            device
        )
        preds = []
        for ex in tqdm(predict_dataset["input_ids"]):
            pred = tokenizer.decode(
                model.generate(
                    torch.tensor([ex]).to(device),
                    max_length=MODEL_INPUT_LENGTH,
                )[0],
                skip_special_tokens=True,
            )
            preds.append(pred)
            # print(pred)
        formatted_preds = []
        for id, pred in zip(ids, preds):
            formatted_preds.append({"id": id, "pred": pred})
        keys = formatted_preds[0].keys()
        if preds_file is None:
            preds_file = "preds.csv"
        print(f"Writing predictions to {preds_file}")
        with open(preds_file, "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(formatted_preds)


if __name__ == "__main__":
    fire.Fire(main)
