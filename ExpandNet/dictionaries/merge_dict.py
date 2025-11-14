import pandas as pd

# --- Load the first dictionary (no headers) ---
df1 = pd.read_csv("en_ur_dict.tsv", sep="\t", header=None, names=["english", "urdu"])

# --- Load the second dictionary (has headers) ---
df2 = pd.read_csv("english_urdu_dictionary_from_tokens.tsv", sep="\t")

# Remove header row from df2 just in case
df2 = df2[df2['english'] != 'english']

# Keep only the two columns, in the correct order
df2 = df2[['english', 'urdu']]

# --- Merge / stack both dictionaries ---
merged = pd.concat([df1, df2], ignore_index=True)

# --- Save without headers ---
merged.to_csv("merged_dictionary.tsv", sep="\t", index=False, header=False)

print("Saved merged_dictionary.tsv with NO HEADERS.")
