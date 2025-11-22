import os
import ast
import nltk
import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    DO NOT PROVIDE ANY OTHER OUTPUT BUT THE URDU WORD!!
    Expected OUTPUT response from you: بوج
    DO NOT REPEAT THE INPUT PROMPT IN YOUR OUPUT ONLY GIVE THE URDU WORD!"""

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
    df = pd.read_csv("data/se_gloss.tsv", sep="\t")
    
    # Create a dictionary with bn_id as key and gloss as value (one-to-one mapping)
    bnid_to_gloss = dict(zip(df['bn_id'], df['gloss']))
    
    urdu_projections = {}
    # for each definition in the senses_dict, we will find the best matching word in Urdu using an LLM
    # and map it to the sense key
    for sense_key, definition in bnid_to_gloss.items():
        print(sense_key, definition)
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
