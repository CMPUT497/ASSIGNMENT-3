import os
import csv
from re import I
import spacy
from collections import defaultdict
from nltk.corpus import wordnet as wn
from langdetect import detect

PREFER_POS_ORDER = (wn.NOUN, wn.ADJ, wn.VERB, wn.ADV)

nlp = spacy.load("en_core_web_lg")

def lemmatize_en(word):
    return nlp(word)[0].lemma_.lower().replace(" ", "_")

def main(): 
    txt_files = []
    tsv_files = []
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dict_dir = f"{root_dir}/dictionaries"
    for fname in os.listdir(dict_dir):
        fpath = os.path.join(dict_dir, fname)
        if os.path.isfile(fpath):
            if fpath.endswith(".txt"):
                txt_files.append(fpath)
            if fpath.endswith(".tsv"):
                tsv_files.append(fpath)

    print(f"Number of .tsv files: {len(tsv_files)}")
    print(f"Number of .txt files: {len(txt_files)}")
    
    rows = {}
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or " " not in line:
                    continue
                ur, en = line.split(" ", 1)
                en = lemmatize_en(en)
                if en not in rows:
                    rows[en] = []
                if ur not in rows[en]:
                    rows[en].append(ur)
    
    for file_path in tsv_files:
        with open(file_path, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                en = lemmatize_en(row[0])
                ur_tokens = row[1].strip().split()
                for ur in ur_tokens:
                    lang_code = detect(ur)
                    if lang_code != 'ur':
                        continue
                    if en not in rows:
                        rows[en] = []
                    if ur not in rows[en]:
                        rows[en].append(ur)

    output_file = f"{root_dir}/dictionaries/combined_dict.tsv"
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for en, ur_list in rows.items():
            ur_str = " ".join(ur_list)
            writer.writerow([en, ur_str])

    print(f"combined_dict.tsv created at {output_file}")


if __name__ == "__main__":
    main()
