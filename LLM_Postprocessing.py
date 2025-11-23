import pandas as pd


def main():
    results = {}
    with open("urdu_projections.tsv", "r", encoding="utf-8") as f:
        data = f.readlines()
        count = 0
        for line in data:
            line = line.split()
            if 'bn:' in line[0]:
                id = line[0]
                count = 0
                results[id] = []
            count += 1
            if count == 5:
                urdu_word = line[-1]
                results[id].append(urdu_word)
        
    with open("LLM_Urdu.tsv", "w", encoding="utf-8") as fout:
        for key, urdu_words in results.items():
            urdu_word = urdu_words
            urdu_word = ', '.join(urdu_word)
            urdu_word = urdu_word[:-1]
            # fout.write(f"{key}\t{urdu_word}\n")
            if urdu_word != "":
                print(key, urdu_word, sep='\t', file=fout)


if __name__ == "__main__":
    main()
