import os
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from openai import OpenAI

# Download WordNet if not already downloaded
try:
    wn.synsets('test')
except LookupError:
    nltk.download('wordnet')
# setting the API key and initializing the client
# OpenAI() will automatically read from OPENROUTER_API_KEY environment variable
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable not set.\n"
    )
client = OpenAI()
model = "gpt-4o-mini"


def get_sense_ids(word, pos):
    # Map POS tags to WordNet format
    pos_map = {
        'NOUN': 'n',
        'VERB': 'v',
        'ADJ': 'a',
        'ADV': 'r'
    }
    wn_pos = pos_map.get(pos)
    # Get synsets for the word
    synsets = wn.synsets(word, pos=wn_pos)
    sense_info = []
    for synset in synsets:
        # Get lemmas for this synset that match the word
        lemmas = [lemma for lemma in synset.lemmas() if lemma.name().lower() == word.lower()]
        # If no matching lemma found, use the first lemma
        if not lemmas:
            lemmas = synset.lemmas()[:1]
        # Get sense key from the matching lemma
        sense_key = lemmas[0].key()
        sense_info.append({
            'sense_key': sense_key,
            'definition': synset.definition(),
        })
    return sense_info


def find_best_matching_word(definition):
    # using the LLM to find the best matching word in Urdu
    prompt = f"""You are a bilingual lexicon expert.
    Given a dictionary definition {definition}, produce the single word in Urdu that best matches this definition. 
    Provide only the Urdu word without explanations! """
    # using the OpenAI API to find the best matching word in Urdu
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    ) 
    return response.choices[0].message.content.strip()


def main():
    # reading the tokens and pos tags from the se13_tokens.tsv file
    df = pd.read_csv("data/se13_tokens.tsv", sep="\t")
    # Filter for only 'instance' type rows
    instance_df = df[df["type"] == "instance"]
    senses_dict = {}
    for _, row in instance_df.iterrows():
        token = row["raw_text"]
        pos_tag = row["pos"]
        lemma = row["lemma"]
        # Get sense IDs for this token
        senses = get_sense_ids(lemma, pos_tag)
        # adding the each sense from synset to the senses_dict         
        for sense in senses:      
            senses_dict[sense['sense_key']] = sense['definition']
    
    # for each definition in the senses_dict, we will find the best matching word in Urdu using an LLM
    for definition in senses_dict.values():
        # using the LLM to find the best matching word in Urdu
        best_matching_word = find_best_matching_word(definition)
        print(best_matching_word)
        break


if __name__ == "__main__":
    main()
