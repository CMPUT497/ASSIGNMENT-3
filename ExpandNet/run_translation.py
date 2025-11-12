import stanza
import re
import pandas as pd
from transformers import pipeline
import xml.etree.ElementTree as ET


def preprocess_text(text):
    text = re.sub(r"-LRB-|-RRB-", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\d+\)", "", text)
    return text

def main():
    nlp = stanza.Pipeline('ur', processors='tokenize,pos,lemma') 
    translator = pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="eng_Latn",
        tgt_lang="urd_Arab"
    )

    xml_path = "./Data/xlwsd_se13.xml"  
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for text_obj in root.findall(".//sentence"):
        sentence_id = text_obj.attrib["id"]
        english_text = " ".join([w.text for w in text_obj.findall("./*")])
        english_text = preprocess_text(english_text)

        ur = translator(english_text)[0]["translation_text"]

        doc = nlp(ur)

        tokens_list = []
        lemmas_list = []
        

        for sent in doc.sentences:
            for word in sent.words:
                tokens_list.append(word.text)
                lemmas_list.append(word.lemma)


        translation_token = " ".join(tokens_list)
        translation_lemma = " ".join(lemmas_list)


        lemma = " ".join(english_text.lower().split())

        rows.append([sentence_id, english_text, ur, lemma, translation_token, translation_lemma])

    df = pd.DataFrame(rows, columns=[
        "sentence_id", "text", "translation", "lemma", "translation_token", "translation_lemma"
    ])

    df.to_csv("expandnet_step1_translate_ur.out.tsv", sep="\t", index=False)

    print("\n Saved expandnet_step1_translate_ur.out.tsv\n")


if __name__ == "__main__":
    main()
