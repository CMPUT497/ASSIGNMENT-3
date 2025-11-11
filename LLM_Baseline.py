import os
import ast
import nltk
import torch
import pandas as pd
from nltk.corpus import wordnet as wn
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download WordNet
nltk.download('wordnet')

# # setting the API key and initializing the client
# load_dotenv()
# # api_key = os.getenv("OPENROUTER_API_KEY")
# api_key = os.getenv("HF_TOKEN")
# if not api_key:
#     raise ValueError(
#         "API_KEY environment variable not set.\n"
#     )
# client = OpenAI(
#     # base_url="https://openrouter.ai/api/v1",
#     base_url="https://router.huggingface.co/v1",
#     api_key=api_key
# )
# model = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# Authenticate with Hugging Face
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)
# initialise the model
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Determine device and dtype
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    device_map = None
else:
    device = "cpu"
    dtype = torch.float32
    device_map = None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=dtype,
    token=hf_token,
    low_cpu_mem_usage=True,
)
# Move model to device
model = model.to(device)


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


def find_best_matching_word(definitions):
    # using the LLM to find the best matching word in Urdu
    prompt = f"""You are a bilingual lexicon expert.
    Given the list of dictionary definitions {definitions}, produce a list of single words in Urdu that best matches each definition in the given definitions list.
    Provide only the Urdu word in the list of Urdu words without explanations for each definition!
    Return a list of Urdu words!"""
    # # using the OpenAI API to find the best matching word in Urdu
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    # )
    # return response.choices[0].message.content.strip()

    # on-device inference with gemma model
    # Tokenize the prompt for model input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    input_length = inputs.input_ids.shape[1]
    # Generate model output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.7,
        )
    # Decode only the generated part (skip the input tokens)
    print(output)
    generated_tokens = output[0][input_length:]
    print(generated_tokens)
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Try to extract a list of Urdu words from the response
    # (Assume model returns a Python-like list or a comma/line separated list as suggested in prompt)
    try:
        # Try parsing if model output a Python list
        urdu_words = ast.literal_eval(response_text)
        print("IT IS A LIST")
        if not isinstance(urdu_words, list):
            urdu_words = [urdu_words]
    except (ValueError, SyntaxError):
        # Fallback: split by common delimiters (comma, newline, or semicolon)
        urdu_words = []
        for delimiter in [",", "\n", ";"]:
            if delimiter in response_text:
                urdu_words = [w.strip() for w in response_text.split(delimiter) if w.strip()]
                break
        # If still no words found, treat the whole response as one word
        if not urdu_words:
            urdu_words = [response_text.strip()] if response_text.strip() else []
    
    return urdu_words


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
    
    urdu_projections = {}
    # for each definition in the senses_dict, we will find the best matching word in Urdu using an LLM
    # and map it to the sense key
    definition_list, key_list = [], []
    batch_size = 0
    for index, (sense_key, definition) in enumerate(senses_dict.items()):
        key_list.append(sense_key)
        definition_list.append(definition)
        batch_size += 1
        # prompt size of 50 definitions per batch
        if batch_size == 50 or index == len(senses_dict)-1:
            best_matching_word = find_best_matching_word(definition_list)
            for i in range(len(key_list)):
                urdu_projections[key_list[i]] = best_matching_word[i]
            definition_list, key_list = [], []
            batch_size = 0
        

    # Create a DataFrame from the dictionary
    urdu_df = pd.DataFrame([
        {'sense_key': sense_key, 'urdu_word': urdu_word}
        for sense_key, urdu_word in urdu_projections.items()
    ])
    
    # saving the urdu projections in a TSV
    output_file = "urdu_projections.tsv"
    urdu_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved {len(urdu_projections)} Urdu projections to {output_file}")


if __name__ == "__main__":
    main()
