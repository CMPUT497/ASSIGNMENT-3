import csv
import spacy

nlp = spacy.load("en_core_web_sm")  

files = [
    "./dictionaries/ur0-en.0-5000.txt",
    "./dictionaries/ur0-en.5000-6500.txt"
]

def normalize_urdu(text):
    return text.replace(" ", "_")

def lemmatize_en(word):
    return nlp(word)[0].lemma_.lower().replace(" ", "_")

rows = [] 

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " " not in line:
                continue

            ur, en = line.split(" ", 1)
            ur = normalize_urdu(ur)
            en = lemmatize_en(en)

            rows.append((en, ur))

output_file = "ur_dict.tsv"
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    for en, ur in rows:
        writer.writerow([en, ur])

print("ur_dict.tsv created")
