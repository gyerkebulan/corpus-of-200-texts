
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TranslationPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка переводчика
print("🧠 Загружаю модель tilmash (KZ → RU)...")
model = AutoModelForSeq2SeqLM.from_pretrained("issai/tilmash").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("issai/tilmash")
tilmash = TranslationPipeline(
    model=model,
    tokenizer=tokenizer,
    src_lang="kaz_Cyrl",
    tgt_lang="rus_Cyrl",
    max_length=1000,
    device=0 if torch.cuda.is_available() else -1
)

def translate_kz_to_ru(text):
    if not text.strip():
        return ""
    return tilmash(text)[0]["translation_text"]

# Настройки
base_url = "https://bilim-all.kz"
category_url_template = "https://bilim-all.kz/article/list/14?page={}"

# Фильтры
religious_keywords = [
    "алла", "құдай", "дін", "намаз", "ораза", "мешіт", "имам", "құран", "хадис",
    "сауап", "кәпір", "тәубе", "дұға", "пайғамбар", "шариғат", "инша", "ислам"
]
additional_block_keywords = [
    "саясат", "үкімет", "партия", "сайлау", "митинг", "қамау", "тергеу", "полиция",
    "өлім", "зорлық", "қылмыс", "сот", "террор", "босқын", "төбелес", "аштық",
    "жемқорлық", "түрме", "қудалау", "радикал", "оппозиция", "революция"
]
gender_keywords = [
    "әйел", "аналар", "келін", "еркек", "ер адам", "күйеуі", "күйеу", "бағыну"
]

block_keywords = religious_keywords + additional_block_keywords + gender_keywords

try:
    df = pd.read_excel("bilim_articles_300_final.xlsx")
    data = df.to_dict("records")
    seen_articles = set((row["title"].lower(), row["paragraph"][:100].lower()) for row in data)
    text_id = max(row["text_id"] for row in data) + 1
    print(f"🔄 Продолжаю с text_id={text_id}, загружено {len(data)} строк.")
except FileNotFoundError:
    data = []
    seen_articles = set()
    text_id = 1
text_id = 1
article_links = []
seen_articles = set()

print("🔍 Сбор ссылок на статьи...")

for page in range(1, 40):
    url = category_url_template.format(page)
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    articles = soup.select("figure figcaption h2 a")
    for a in articles:
        full_url = base_url + a['href']
        article_links.append(full_url)
        if len(article_links) >= 1000:
            break
    if len(article_links) >= 1000:
        break
    time.sleep(0.3)

print(f"✅ Найдено {len(article_links)} ссылок.\n")

MAX_ARTICLES = 215

for url in article_links:
    if text_id > MAX_ARTICLES:
        break
    if text_id > MAX_ARTICLES:
        break
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        title_tag = soup.select_one("div.blogtext h1.heading")
        title = title_tag.get_text(strip=True) if title_tag else "Без названия"
        if 'көк сөз' in title.lower():
            print(f"⛔️ Пропущено (көк сөз в названии): {title}")
            continue
        title_ru = translate_kz_to_ru(title)

        author_tag = soup.select_one("div.blogmetas li a[href^='/user/profile/']")
        author = author_tag.get_text(strip=True) if author_tag else "Неизвестен"

        category_tags = soup.select("div.blogmetas li i.fa-align-justify ~ a")
        subcategory = category_tags[1].get_text(strip=True) if len(category_tags) > 1 else "Не указано"
        subcategory_ru = translate_kz_to_ru(subcategory)

        paragraphs = soup.select("div.blogtext p")
        clean_paragraphs = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]

        if not clean_paragraphs:
            continue

        key = (title.lower(), clean_paragraphs[0][:100].lower())
        if key in seen_articles:
            print(f"⛔️ Пропущено (дубликат): {title}")
            continue
        seen_articles.add(key)

        text_combined = " ".join(clean_paragraphs).lower()
        matched_words = [word for word in block_keywords if word in text_combined]
        if matched_words:
            print(f"⛔️ Пропущено (религия / негатив / гендер): {title} → Слова: {', '.join(matched_words)}")
            continue
            print(f"⛔️ Пропущено (религия / негатив / гендер): {title}")
            continue

        print(f"📄 [{text_id}] {title} — абзацев: {len(clean_paragraphs)}")

        for paragraph in clean_paragraphs:
            if "әлеуметтік желілерде" not in paragraph.lower():
                paragraph_ru = translate_kz_to_ru(paragraph)
                data.append({
                    "text_id": text_id,
                    "paragraph": paragraph,
                    "p_ru": paragraph_ru,
                    "title": title,
                    "title_ru": title_ru,
                    "author": author,
                    "url": url,
                    "category": "Бала тәрбиесі",
                    "subcategory": subcategory,
                    "subcategory_ru": subcategory_ru
                })

        text_id += 1
        if text_id % 10 == 0:
            print("💾 Промежуточное сохранение...")
            df = pd.DataFrame(data)
            df.to_excel("bilim_articles_300_final.xlsx", index=False)
        time.sleep(0.2)

    except Exception as e:
        print(f"⚠️ Ошибка в статье {url}: {e}")

df = pd.DataFrame(data)
df.to_csv("bilim_articles_300_final.csv", index=False, encoding="utf-8-sig")
df.to_excel("bilim_articles_300_final.xlsx", index=False)

print(f"\n✅ Готово!")
print(f"🔸 Сохранено строк (абзацев): {len(df)}")
print("🔸 CSV: bilim_articles_300_final.csv")
print("🔸 Excel: bilim_articles_300_final.xlsx")
