from typing import List

import openai
import pandas as pd
from retry import retry
from tqdm import tqdm

k = 10


@retry(Exception, delay=30, tries=-1)
def generate_segs_and_gloss(sents: List[str]) -> List[str]:
    outputs = []
    for sent in tqdm(sents):
        prompt = f"Use the Leipzig glossing rules to generate segmentation and interlinear gloss for the following Russian sentence: {sent}\n\nYour response should be in the following format:\nSentence: <Russian sentence>\nSegmentation: <Segmented Russian sentence>\nGloss: <Glossed sentence>\n\nMorphemes should be separated by dash ('-').  The number of gloss labels and segments should be the same. Do not include extra notes or information in the output."  # noqa: E501
        print(prompt)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        outputs += [completion.choices[0].message.content]
        print(outputs)
    return outputs


def main():
    openai.api_key = open("openAI_api.key").read()
    data_path = "data/eng-rus-sorted.csv"
    df = pd.read_csv(data_path)
    top_df = df.nlargest(k, "similarity", keep="all")
    rus_sents = top_df["rus"].tolist()
    generated_text = generate_segs_and_gloss(rus_sents)
    top_df["generated"] = generated_text
    top_df.to_csv("data/eng-rus-outputs.csv", index=False)


if __name__ == main():
    main()
