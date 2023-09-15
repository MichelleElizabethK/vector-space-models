from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk.tag import pos_tag

import nltk
import string
import simplemma


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class Preprocess:

    def __init__(self) -> None:
        self.text = ''
        self.stop_words = {
            'en': stopwords.words('english'),
            'cs': []
        }
        with open ("czech_stopwords.txt", "r") as f:
            self.stop_words['cs'] = f.read().splitlines()

    def preprocess(self, text, lang, run, lower_case=False, remove_stopwords=False, stem=False, lemmatize=False) -> list:
        self.text = text
        if lower_case:
            self.text = self.text.lower()
        if run == 'run-0':
            words = self.text.split()
        else:
            words = word_tokenize(self.text) 
        table = str.maketrans('', '', string.punctuation)
        tokenized_text = [w.translate(table) for w in words]
        tokenized_text = [token for token in tokenized_text if len(token) > 0]
        if lemmatize:
            tokenized_text = self.perform_lemmatization(tokenized_text, lang)
        if stem:
            tokenized_text = self.perform_stemming(tokenized_text, lang)
        if remove_stopwords:
            tokenized_text = self.remove_stopwords(tokenized_text, lang)
        return tokenized_text

    def remove_stopwords(self, tokenized_text, lang) -> list:
        tokens_without_stopwords = [
            word for word in tokenized_text if word not in self.stop_words[lang]]
        return tokens_without_stopwords

    def perform_stemming(self, tokenized_text, lang) -> list:
        if lang == 'en':
            stemmer = SnowballStemmer('english')
            # stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokenized_text]
        return tokens

    def perform_lemmatization(self, tokens, lang) -> list:
        # if lang == 'en':
        #     token_with_tags = pos_tag(tokens)
        #     lemmatizer = WordNetLemmatizer()
        #     lemmatized_text = [lemmatizer.lemmatize(token[0], 'n') if token[1].startswith("NN") else lemmatizer.lemmatize(
        #         token[0], 'v') if token[1].startswith("VB") else lemmatizer.lemmatize(token[0], 'a') for token in token_with_tags]
        # else:
        lemmatized_text = [simplemma.lemmatize(token, lang) for token in tokens]

        return lemmatized_text

