import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from googletrans import Translator
import re

# Ensure consistent language detection
DetectorFactory.seed = 0

def is_singlish(text):
    # Heuristic: common Sinhala words written in Latin script
    singlish_keywords = [
        'wada', 'weda', 'nehe', 'nhe', 'wadkma',
    ]
    text_lower = text.lower()
    for word in singlish_keywords:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            return True
    return False

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'unknown'

def main():
    df = pd.read_csv('data/slt_selfcare_google_reviews.csv')
    translator = Translator()
    detected_langs = []
    is_singlish_list = []
    translations = []

    for review in df['review_description'].fillna(''):
        lang = detect_language(review)
        detected_langs.append(lang)
        singlish = False
        if lang == 'si':
            # Sinhala detected
            try:
                translated = translator.translate(review, src='si', dest='en').text
            except Exception:
                translated = ''
        elif lang == 'en' and is_singlish(review):
            singlish = True
            try:
                translated = translator.translate(review, src='auto', dest='en').text
            except Exception:
                translated = ''
        else:
            translated = review
        is_singlish_list.append(singlish)
        translations.append(translated)

    df['detected_language'] = detected_langs
    df['is_singlish'] = is_singlish_list
    df['translated_review'] = translations
    df.to_csv('data/slt_reviews_translated.csv', index=False)
    print('✅ Reviews processed and saved to slt_reviews_translated.csv')

if __name__ == '__main__':
    main()
