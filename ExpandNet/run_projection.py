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
                      help="Use a dictionary for filtering. Available options: none, bn (BabelNet), wik (WiktExtract), wikpan (WiktExtract and PanLex)")
  parser.add_argument("--alignment_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="File containing the output of step 2 (alignment).")
  parser.add_argument("--output_file", type=str, default="expandnet_step3_project.out.tsv")
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
print(f"Dataset loaded: {len(df_src)} rows")

print("Loading alignment data...")
df_sent = pd.read_csv(args.alignment_file, sep='\t')
print(f"Alignment loaded: {len(df_sent)} sentences")

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
        return False
    return ur_word in dict_[eng_word]

def get_alignments(alignments, i):
    """Get all target indices aligned to source index i."""
    return [link[1] for link in alignments if link[0] == i]

# Load the dictionary.
print("Loading dictionary...")
dict_wik = load_dict([args.dictionary])
print(f"Dictionary loaded")

# Group by sentence_id and aggregate bn_gold and lemma values into lists
print("Preparing data...")
# Filter out empty strings and NaN values before grouping
df_src_filtered = df_src[df_src['bn_gold'].notna() & (df_src['bn_gold'] != '')].copy()
print(f"Rows in df_src_filtered: {len(df_src_filtered)}")

bn_gold_lists = (
    df_src_filtered.groupby("sentence_id")["bn_gold"]
       .apply(list)
       .reset_index(name="bn_gold")
)

lemma_gold_lists = (
    df_src.groupby("sentence_id")["lemma"]
       .apply(list)
       .reset_index(name="lemma_gold")
)

print("Length of bn_gold_lists:", len(bn_gold_lists))
print("Length of lemma_gold_lists:", len(lemma_gold_lists))

# Merge back into df_sent
df_sent = (
    df_sent.merge(bn_gold_lists, on="sentence_id", how="left")
           .merge(lemma_gold_lists, on="sentence_id", how="left")
)

print("Length of df_sent:", len(df_sent))

print(f"Data prepared")

# Project senses
print("Projecting senses...")
senses = set()
for _, row in df_sent.iterrows():
    src = row['lemma_gold']
    tgt = row['translation_lemma'].split(' ')
    print("Target lemma:", tgt)
    ali = ast.literal_eval(row['alignment'])
    print("Alignment:", ali)
    bns = row['bn_gold']
    print(bns)
    print()
  

    # Handle NaN or non-list values
    # Check if bns is NaN (avoid ambiguity with arrays)
    try:
        is_na = pd.isna(bns)
        if hasattr(is_na, 'any'):
            if is_na.any():
                continue
            elif is_na:
                continue
    except (TypeError, ValueError):
        if bns is None or (isinstance(bns, float) and pd.isna(bns)):
            continue
    
    # Convert to list if needed
    if not isinstance(bns, list):
        if hasattr(bns, 'tolist'):
            bns = bns.tolist()
        elif hasattr(bns, '__iter__') and not isinstance(bns, str):
            bns = list(bns)
        else:
            continue
    
    if not isinstance(src, list):
        if hasattr(src, 'tolist'):
            src = src.tolist()
        elif hasattr(src, '__iter__') and not isinstance(src, str):
            src = list(src)
        else:
            continue

    for i, bn in enumerate(bns):
        if not str(bn)[:3] == 'bn:':
            continue
        alignment_indices = get_alignments(ali, i)
        print("alignment indices:", alignment_indices)
        if len(alignment_indices) > 1:
            candidates = [args.join_char.join([tgt[j] for j in alignment_indices])]
        elif len(alignment_indices) == 1:
            candidates = [tgt[alignment_indices[0]]]
        else:
            candidates = []
        print("Candidates:", candidates)
        if candidates:
            for candidate in candidates:
                source = src[i]
                print("src:", src)
                if is_valid_translation(source, candidate, dict_wik):
                    print(f"CHECK: Valid translation found for source '{source}' and candidate '{candidate}' with BabelNet synset '{bn}'")
                    senses.add((bn, candidate))

print()


print(f"Found {len(senses)} unique sense-lemma pairs")

print(f"Saving results to {args.output_file}...")
with open(args.output_file, 'w') as f:
  for (bn, lemma) in sorted(senses):
    print(bn, lemma, sep='\t', file=f)

print('Complete!')