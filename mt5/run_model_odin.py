import pathlib

# from data import load_data_file, prepare_dataset, write_predictions
import datasets
import evaluate
import fire
import numpy as np
import torch
from tqdm import tqdm
import transformers
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
    prompt2 = ", with its {transl_lang} tranlation: "

    def collate_fn(batch):
        nonlocal tokenizer, prompt1, prompt2, max_length
        inputs = [
            prompt1.format(src_lang=lang) + _reassemble(ex)
            for lang, ex in zip(batch["language"], batch["transcription"])
        ]
        if "translation" in batch and use_translations:
            for i, ex in enumerate(batch["translation"]):
                inputs[i] = (
                    inputs[i] + prompt2.format(transl_lang="English") + _reassemble(ex)
                )
        inputs = [t + "\nAnswer: " for t in inputs]

        if "gloss" in batch:
            targets = [" ".join(ex) for ex in batch["gloss"]]
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
        # evaluation_strategy="epoch",
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
        logging_steps=100,
        tf32=True,
    )

    trainer = transformers.Trainer(
        model,
        args,
        optimizers=[optimizer, lr_scheduler],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100
        ),
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
    )
    return trainer


def gather_odin_data(odin_root):
    print("Getting odin language names from glottolog")
    from pyglottolog import Glottolog

    glottolog = Glottolog("./glottolog")
    languages = dict()
    oroot = pathlib.Path(odin_root)
    for file in tqdm(oroot.glob("*.txt")):
        langcode = file.stem
        lang = glottolog.languoid(langcode.split("-")[0])
        if lang is not None:
            languages[langcode] = lang.name
    return languages


def main(
    mode: str,
    odin_root: str,
    model_path: str,
):
    if mode == "train":
        wandb.init(project="sigmorphon2023", entity="wav2gloss")

    MODEL_INPUT_LENGTH = 1024

    pretrained_model = "google/byt5-base"

    if mode == "train":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model, use_fast=False
        )

        dataset = datasets.load_dataset(odin_root, "all")
        collator = get_collator(tokenizer, max_length=MODEL_INPUT_LENGTH)
        dataset["train"] = dataset["train"].shuffle()
        dataset["train"] = dataset["train"].map(
            collator, batched=True, remove_columns=dataset["train"].column_names
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
        )

        print("Training...")
        trainer.train()
        print(f"Saving model to {model_path}")
        trainer.save_model(model_path)
        print(f"Model saved at {model_path}")
    elif mode == "predict":
        pass
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     pretrained_model, use_fast=False
        # )
        # if lang == "all":
        #     langs = list(languages.keys())
        # else:
        #     langs = [lang]
        # for lang in langs:
        #     data_path = (
        #         f"../data/{languages[lang]}/{lang}-dev-track{track_num}-uncovered"
        #     )
        #     predict_data = load_data_file(data_path)
        #     predict_dataset = prepare_dataset(data=predict_data)
        #     collator = get_collator(
        #         tokenizer=tokenizer,
        #         src_lang=languages[lang],
        #         transl_lang=translations[lang],
        #         max_length=MODEL_INPUT_LENGTH,
        #     )
        #     predict_dataset = predict_dataset.map(
        #         collator, batched=True, remove_columns=predict_dataset.column_names
        #     )
        #     model = transformers.AutoModelForPreTraining.from_pretrained(
        #         model_path).to(device)

        #     preds = []
        #     for ex in tqdm(predict_dataset["input_ids"]):
        #         preds.append(
        #             tokenizer.decode(
        #                 model.generate(
        #                     torch.tensor([ex]).to(device),
        #                     max_length=MODEL_INPUT_LENGTH,
        #                 )[0],
        #                 skip_special_tokens=True,
        #             )
        #         )

        #     write_predictions(
        #         data_path,
        #         lang=lang,
        #         decoded_preds=preds,
        #     )


if __name__ == "__main__":
    fire.Fire(main)
