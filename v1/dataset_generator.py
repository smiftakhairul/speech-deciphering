import pandas as pd
import random
import nltk
import os
from googletrans import Translator

class DatasetGenerator:
    def __init__(self, languages={'bn': 'Bengali', 'ru': 'Russian'}):
        self.vendor_dir = "./vendor"
        self.languages = languages
        if not os.path.exists(os.path.join(self.vendor_dir, 'nltk_data/corpora/brown')):
            nltk.download('brown', download_dir=os.path.join(self.vendor_dir, 'nltk_data'))
        nltk.data.path.append(os.path.join(self.vendor_dir, 'nltk_data'))
        self.translator = Translator()

    def generate_random_english_sentence(self):
        corpus = nltk.corpus.brown.sents()
        sentence = ' '.join(random.choice(corpus))
        words = sentence.split()
        simplified_sentence = ' '.join(words[:5])
        return simplified_sentence

    def translate_to_language(self, text, dest_language):
        translation = self.translator.translate(text, dest=dest_language)
        return translation.text

    def generate_and_translate(self, num_rows=2):
        data = []
        for _ in range(num_rows):
            english_sentence = self.generate_random_english_sentence()
            translations = [self.translate_to_language(english_sentence, lang) for lang in self.languages.keys()]
            data.append([english_sentence] + translations)

        columns = ['English'] + list(self.languages.values())
        df = pd.DataFrame(data, columns=columns)
        return df
