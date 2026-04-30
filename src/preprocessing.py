import pandas as pd
import re
import string

def load_data():

    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0)

    df = df[["text", "label"]]

    return df


def clean_text(text):

    text = text.lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return text