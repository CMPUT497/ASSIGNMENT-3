import os
import csv
from re import I
import spacy
from collections import defaultdict
from nltk.corpus import wordnet as wn

PREFER_POS_ORDER = (wn.NOUN, wn.ADJ, wn.VERB, wn.ADV)

nlp = spacy.load("en_core_web_lg")

def normalize_urdu(text):
    return text.replace(" ", "_")

def lemmatize_en(word):
    return nlp(word)[0].lemma_.lower().replace(" ", "_")

def main(): 
    files = []
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dict_dir = f"{root_dir}/dictionaries"
    for fname in os.listdir(dict_dir):
        fpath = os.path.join(dict_dir, fname)
        if os.path.isfile(fpath):
            if fpath.endswith(".txt"):
                files.append(fpath)

    rows = {}
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or " " not in line:
                    continue
                ur, en = line.split(" ", 1)
                ur = normalize_urdu(ur)
                en = lemmatize_en(en)
                if en not in rows:
                    rows[en] = []
                if ur not in rows[en]:
                    rows[en].append(ur)

    output_file = f"{root_dir}/dictionaries/en_ur_dict.tsv"
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for ur, en_list in rows.items():
            en_str = " ".join(en_list)
            writer.writerow([ur, en_str])

    print(f"en_ur_dict.tsv created at {output_file}")


if __name__ == "__main__":
    main()
