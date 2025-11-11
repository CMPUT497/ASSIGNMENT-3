import scrapy
from scrapy.crawler import CrawlerProcess
import csv
import re

def clean_english(raw):
    raw = raw.lower().strip()


    raw = re.sub(r"\(.*?\)", "", raw)


    raw = raw.replace("'", "").replace('"', '')


    parts = re.split(r"[;,]", raw)

    cleaned = []
    for word in parts:
        word = word.strip().replace(" ", "_")

        word = re.sub(r"^(a|an|the)_", "", word)

        word = re.sub(r"^being_", "", word)

        word = word.lstrip("_")

        if len(word.split("_")) > 4:
            continue

        if any(char.isdigit() for char in word):
            continue

        if word == "" or word in {"_", "-"}:
            continue

        cleaned.append(word)

    return cleaned


def clean_urdu(raw):
    raw = raw.strip()

    raw = re.sub(r"\[\d+\]", "", raw)
    raw = re.sub(r"\(\d+\)", "", raw)

    raw = raw.replace("'", "").replace('"', '')

    raw = re.sub(r"\s+", " ", raw).strip()

    return raw


output_file = "urdu_eng_dict.tsv"

en_to_ur = {}

class ScrapUrduWords(scrapy.Spider):
    name = "Urdu Dictionary Spider"

    def start_requests(self):
        for i in range(1, 118364):   
            yield scrapy.Request(
                url=f"http://urdulughat.info/words/{i}",
                callback=self.parse,
                dont_filter=True
            )

    def parse(self, response):
        urdu_word = response.css("h2::text").get()
        if not urdu_word:
            return

        urdu_word = clean_urdu(urdu_word)
        translations = response.css("div#english-translations > ul > li::text").getall()

        for raw_en in translations:
            for en in clean_english(raw_en):
                if en not in en_to_ur:
                    en_to_ur[en] = set()
                en_to_ur[en].add(urdu_word)


process = CrawlerProcess()
process.crawl(ScrapUrduWords)
process.start()

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    for en_word, ur_set in sorted(en_to_ur.items()):
        for ur_word in sorted(ur_set):
            writer.writerow([en_word, ur_word])

print(f"Saved cleaned dictionary as: {output_file}")
