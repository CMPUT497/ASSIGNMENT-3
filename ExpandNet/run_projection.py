import argparse
import ast
import csv
import pandas as pd
import sys
import xml_utils

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="xlwsd_se13.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--src_gold", type=str, default="xlwsd_se13.key.txt",
                      help="Path to the gold sense tagging file.")
  parser.add_argument("--dictionary", type=str, default="wikpan-en-fr.tsv",
                      help="Use a dictionary for filtering. Available options: none, wn_sense (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--alignment_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="File containing the output of step 2 (alignment).")
  parser.add_argument("--output_file", type=str, default="expandnet_step3_project.out.tsv")
  parser.add_argument("--token_info_file", type=str, default="expandnet_step3_project.token_info.tsv",
                      help="(Helpful for understanding the process undergone.)")
  parser.add_argument("--join_char", type=str, default='')
  return parser.parse_args()

args = parse_args()

# Set CSV field size limit to a large but safe value for Windows compatibility
# sys.maxsize can be too large on Windows (32-bit int limit)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # On Windows, use a large but safe value (2^31 - 1)
    csv.field_size_limit(2147483647)

print(f"Source data:     {args.src_data}")
print(f"Source gold:     {args.src_gold}")
print(f"Dictionary:      {args.dictionary}")
print(f"Alignment file:  {args.alignment_file}")
print(f"Output file:     {args.output_file}")

# Load the dataset and alignment data.
print("Loading dataset...")
df_src = xml_utils.process_dataset(args.src_data, args.src_gold)

print("Loading alignment data...")
df_sent = pd.read_csv(args.alignment_file, sep='\t')

def load_dict(filepaths):
    """Load multiple TSV files into a dict: {english_word: set(french_words)}.
    All spaces are normalized to underscores.
    """
    dict_ = {}
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for line_num, row in enumerate(reader, start=1):
                if len(row) < 2:
                    print(f"Warning: Line {line_num} in {filepath} has fewer than 2 columns.")
                    continue
                eng_word = row[0].strip().lower().replace(' ', '_')  # Normalize English key
                ur_words = set(word.strip().lower().replace(' ', '_') for word in row[1].split())
                if eng_word in dict_:
                    dict_[eng_word].update(ur_words)  # Merge sets if key exists
                else:
                    dict_[eng_word] = ur_words
    return dict_

def is_valid_translation(eng_word, ur_word, dict_):
    """Check if (eng_word, ur_word) is a valid translation pair in the dict."""
    eng_word = eng_word.lower().strip().replace(' ', '_')
    ur_word = ur_word.lower().strip().replace(' ', '_')
    if eng_word not in dict_:
        return True
    return ur_word in dict_[eng_word]

def get_alignments(alignments, i):
    """Get all target indices aligned to source index i."""
    return [link[1] for link in alignments if link[0] == i]

# Load the dictionary.
print("Loading dictionary...")
dict_wik = load_dict([args.dictionary])
print(f"Dictionary loaded")

print("Preparing data...")
# Filter out empty strings and NaN values before grouping
df_src_filtered = df_src[df_src['gold'].notna() & (df_src['gold'] != '')].copy()

# Group by sentence_id and aggregate gold and lemma values into lists
gold_lists = (
    df_src_filtered.groupby("sentence_id")["gold"]
       .apply(list)
       .reset_index(name="gold")
)

lemma_gold_lists = (
    df_src.groupby("sentence_id")["lemma"]
       .apply(list)
       .reset_index(name="lemma_gold")
)

token_gold_lists = (
    df_src.groupby("sentence_id")["text"]
       .apply(list)
       .reset_index(name="token_gold")
)

# Merge back into df_sent
df_sent = (
    df_sent.merge(gold_lists, on="sentence_id", how="left")
           .merge(lemma_gold_lists, on="sentence_id", how="left")
           .merge(token_gold_lists, on="sentence_id", how="left")
)
print(f"Data prepared")

# Project senses
print("Projecting senses...")
senses = set()
with open(args.token_info_file, 'w', encoding='utf-8') as f:
    f.write("Token ID" + '\t' + "Source Token" + '\t' + "Source Lemma" + '\t' + "Source POS" + '\t' + "Translated Token" + '\t'  + "Translated Lemma" + '\t' + "Synset ID" + '\t' + "Link in Dictionary?" + '\n')

    for _, row in df_sent.iterrows():
        tok_num = 0
        src = row['lemma_gold']
        src_tok = row['token_gold']
        assert len(src) == len(src_tok)
        
        tgt = row['translation_lemma'].split(' ')
        tgt_tok = row['translation_token'].split(' ')
        assert len(tgt) == len(tgt_tok)
        
        ali = ast.literal_eval(row['alignment'])
        gold_senses = row['gold']
        sent_id = row['sentence_id']
        
        # Handle NaN or non-list values
        # Check if gold_senses is NaN (avoid ambiguity with arrays)
        try:
            is_na = pd.isna(gold_senses)
            if hasattr(is_na, 'any'):
                # It's an array/Series, check if any element is NA
                if is_na.any():
                    continue
            else:
                # It's a scalar, check directly
                if is_na:
                    continue
        except (TypeError, ValueError):
            if gold_senses is None or (isinstance(gold_senses, float) and pd.isna(gold_senses)):
                continue
        
        # Convert to list if needed
        if not isinstance(gold_senses, list):
            if hasattr(gold_senses, 'tolist'):
                gold_senses = gold_senses.tolist()
            elif hasattr(gold_senses, '__iter__') and not isinstance(gold_senses, str):
                gold_senses = list(gold_senses)
            else:
                continue
        
        if not isinstance(src, list):
            if hasattr(src, 'tolist'):
                src = src.tolist()
            elif hasattr(src, '__iter__') and not isinstance(src, str):
                src = list(src)
            else:
                continue

        for i, wn_sense in enumerate(gold_senses):
            sense_token = wn_sense.split('%')[0]
            # if '_' in sense_token:
            #     print("THIS CONTAINS MULTIPLE WORD TOKEN", sense_token)
            idx = src.index(sense_token)
            source = src[idx]
            tok = src_tok[idx]
            tok_id = sent_id + f".s{tok_num:03d}"
            tok_num += 1
            
            alignment_indices = get_alignments(ali, idx)
            if len(alignment_indices) > 1:
                candidates = [args.join_char.join([tgt[j] for j in alignment_indices])]
                t_candidates = [args.join_char.join([tgt_tok[j] for j in alignment_indices])]
            elif len(alignment_indices) == 1:
                candidates = [tgt[alignment_indices[0]]]
                t_candidates = [tgt_tok[alignment_indices[0]]]
            else:
                candidates = []
                t_candidates = []
            
            if candidates:
                for t_candidate, candidate in zip(t_candidates, candidates):
                    
                    src_pos = wn_sense[-1].upper()
                    f.write(tok_id + '\t' + tok + '\t' + source + '\t' + src_pos + '\t' + t_candidate + '\t'  + candidate + '\t' + wn_sense + '\t' + str(is_valid_translation(source, candidate, dict_wik)) + '\n')
                    # easing this check to only checking if the english lemmas are present in the dictionary
                    if source in dict_wik:    
                    # if is_valid_translation(source, candidate, dict_wik):            
                        senses.add((wn_sense, candidate))
                    else:
                        # print(f"{sent_id}, {idx}, {alignment_indices}, {source} is NOT present in the dictionary for Urdu Word {candidate}: {wn_sense}")
                        print(f"{source} is NOT present in the dictionary for Urdu Word {candidate}: {wn_sense}")


print(f"Found {len(senses)} unique sense-lemma pairs")

print(f"Saving results to {args.output_file}...")
with open(args.output_file, 'w', encoding='utf-8') as f:
  for (wn_sense, lemma) in sorted(senses):
    print(wn_sense, lemma, sep='\t', file=f)

print('Complete!')
