# script to sort and extract top k similar sentences to training data
# outputs a csv file of sentences for augmented training data
from typing import List

from data import load_data_file
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

target_lang = "lez"
augment_lang = "rus"
trans_lang = "eng"
k = 1000

languages = {
    "arp": "Arapaho",
    "git": "Gitksan",
    "lez": "Lezgi",
    "nyb": "Nyangbo",
    "ddo": "Tsez",
    "usp": "Uspanteko",
}

device = "mps" if torch.has_mps else "cpu"
print(device)
torch.cuda.empty_cache()


def get_avg_translation_emb(path: str) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    data = load_data_file(path)
    translations = [x.translation for x in data]
    embeddings = model.encode(
        translations, show_progress_bar=True, convert_to_numpy=True
    )
    all_embeddings = np.vstack(embeddings)
    print(all_embeddings.shape)
    mean_embedding = np.mean(all_embeddings, axis=0)
    print(mean_embedding.shape)
    return mean_embedding


def get_cos_similarities(translations: List[str], mean_emb: np.ndarray) -> List[float]:
    """Given a list of translations from bitext data (from dataframe),
    return cosine similarities for each"""
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(
        translations, show_progress_bar=True, convert_to_numpy=True
    )
    cos_similarities = []
    for embedding in tqdm(embeddings):
        cos_sim = util.cos_sim(embedding, mean_emb)
        cos_similarities += [cos_sim.item()]
    return cos_similarities


def main():
    target_lang_name = languages[target_lang]
    target_lang_path = (
        f"../data/{target_lang_name}/{target_lang}-train-track2-uncovered"
    )
    aug_data_path = f"data/{trans_lang}-{augment_lang}.csv"
    out_path = f"data/{trans_lang}-{augment_lang}-sorted.csv"
    df_aug_data = pd.read_csv(aug_data_path)
    mean_emb = get_avg_translation_emb(target_lang_path)
    df_aug_data["similarity"] = get_cos_similarities(
        df_aug_data[trans_lang].astype(str).tolist(), mean_emb
    )
    df_aug_data.sort_values("similarity", ascending=False, inplace=True)
    df_aug_data.to_csv(out_path, index=False)


if __name__ == main():
    main()
