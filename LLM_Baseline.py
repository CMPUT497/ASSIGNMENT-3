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


def find_best_matching_word(definition):
    TARGET_LANGUAGE = "Urdu"
    # using the LLM to find the best matching word in Urdu
    prompt = f"""You are a bilingual lexicon expert.
    Given a dictionary definition: "{definition}", produce the single word in {TARGET_LANGUAGE} that best matches this definition. 
    Provide only the one {TARGET_LANGUAGE} word without explanations!
    DO NOT PROVIDE ANY OTHER OUPUT BUT THE URDU WORD!!
    Example (Do not include OUTPUT in your response, here INPUT and OUTPUT are only present to help you distinguish INPUT and OUTPUT, they should not be present in the your response), 
    (Only the urdu word must be present in your response)
    Given INPUT prompt: You are a bilingual lexicon expert.
    Given a dictionary definition: "burden", produce the single word in {TARGET_LANGUAGE} that best matches this definition. 
    Provide only the one {TARGET_LANGUAGE} word without explanations!
    DO NOT PROVIDE ANY OTHER OUPUT BUT THE URDU WORD!!
    Expected OUTPUT response from you: بوج"""
    # # using the OpenAI API to find the best matching word in Urdu
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ],
    # )
    # return response.choices[0].message.content.strip()

    # on-device inference with gemma model, expecting a single word per prompt/response
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
    generated_tokens = output[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response_text


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
    for sense_key, definition in senses_dict.items():
        best_matching_word = find_best_matching_word(definition)
        urdu_projections[sense_key] = best_matching_word

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
