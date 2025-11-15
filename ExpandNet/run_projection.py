import argparse
import ast
import csv
import pandas as pd
import sys
import xml_utils

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
  print(f"Searching the words {eng_word} and {ur_word}")
  if eng_word not in dict_:
    return True
  return ur_word in dict_[eng_word]

def get_alignments(alignments, i):
  """Get all target indices aligned to source index i."""
  return [link[1] for link in alignments if link[0] == i]

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--src_data", type=str, default="xlwsd_se13.xml",
                      help="Path to the XLWSD XML corpus file.")
  parser.add_argument("--src_gold", type=str, default="xlwsd_se13.key.txt",
                      help="Path to the gold sense tagging file.")
  parser.add_argument("--dictionary", type=str, default="wikpan-en-fr.tsv",
                      help="Use a dictionary for filtering. Available options: none, bn (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--alignment_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="File containing the output of step 2 (alignment).")
  parser.add_argument("--output_file", type=str, default="expandnet_step3_project.out.tsv")
  parser.add_argument("--token_info_file", type=str, default="expandnet_step3_project.token_info.tsv",
                      help="(Helpful for understanding the process undergone.)")
  parser.add_argument("--join_char", type=str, default='_')
  return parser.parse_args()

args = parse_args()

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
print(f"Dataset loaded: {len(df_src)} rows")
# Write df_src to a JSON file
# df_src.to_json("df_src.json", orient="records", force_ascii=False, lines=True)

print("Loading alignment data...")
df_sent = pd.read_csv(args.alignment_file, sep='\t')
print(f"Alignment loaded: {len(df_sent)} sentences")
# Write df_sent to a JSON file
# df_sent.to_json("df_sent.json", orient="records", force_ascii=False, lines=True)

# Load the dictionary.
print("Loading dictionary...")
dict_wik = load_dict([args.dictionary])
print(f"Dictionary loaded")

# Group by sentence_id and aggregate bn_gold and lemma values into lists
print("Preparing data...")
bn_gold_lists = (
    df_src.groupby("sentence_id")["bn_gold"]
       .apply(list)
       .reset_index(name="bn_gold")
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
    df_sent.merge(bn_gold_lists, on="sentence_id", how="left")
           .merge(lemma_gold_lists, on="sentence_id", how="left")
           .merge(token_gold_lists, on="sentence_id", how="left")
)
print(f"Data prepared!")

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
        
        print("src:", src)
        print("tgt:", tgt)

        ali = ast.literal_eval(row['alignment'])
        bns = row['bn_gold']
        sent_id = row['sentence_id']
        print("bns:", bns)
        
        for i, bn in enumerate(bns):
            source = src[i]
            tok = src_tok[i]
            tok_id = sent_id + f".t{tok_num:03d}"
            if not str(bn)[:3] == 'bn:':
                f.write('wf' + '\t' + tok + '\t' + source + '\t' + ' ' + '\t'  + ' ' + '\t' + ' ' + '\t' + ' ' + '\n')
                continue
            tok_num += 1
            alignment_indices = get_alignments(ali, i)
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
                    src_pos = bn[-1].upper()
                    f.write(tok_id + '\t' + tok + '\t' + source + '\t' + src_pos + '\t' + t_candidate + '\t'  + candidate + '\t' + bn + '\t' + str(is_valid_translation(source, candidate, dict_wik)) + '\n')
                    if is_valid_translation(source, candidate, dict_wik):
                        senses.add((bn, candidate))
        break

print(f"Found {len(senses)} unique sense-lemma pairs")

print(f"Saving results to {args.output_file}...")
with open(args.output_file, 'w') as f:
  for (bn, lemma) in sorted(senses):
    print(bn, lemma, sep='\t', file=f)

print('Complete!')
