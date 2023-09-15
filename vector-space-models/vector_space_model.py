from preprocess import Preprocess
from bs4 import BeautifulSoup
from multiprocessing.pool import Pool
from multiprocessing import Manager
from itertools import chain
from operator import methodcaller
import lxml
import re
import numpy as np
from tqdm import tqdm

from config import config

import warnings
warnings.filterwarnings(action='once')


class VectorSpaceModel:

    def __init__(self) -> None:
        self.preprocess = Preprocess()
        self.tags = {
            'en': ['DD', 'HD', 'DH', 'AU', 'BD', 'LD', 'TE', 'DC'],
            'cs': ['TITLE', 'HEADING', 'GEOGRAPHY', 'TEXT']
        },
        self.selected_tags = []
        self.doc_list = ''
        self.parsed_doc_list = Manager().dict()
        self.doc_length = Manager().dict()
        self.topics = ''
        self.doc_index = []
        self.lang = ''
        self.doc_path = 'documents_'
        self.inverted_index_dict = {}
        self.doc_norm = {}
        self.queries = []
        self.topic_title_map = {}
        self.term_topic_index = {}
        self.final_ranking = {}
        self.doc_term_weighting = ''
        self.doc_weighting = ''
        self.query_term_weighting = ''
        self.query_weighting = ''
        self.normalization = ''
        self.inverted_index_dict_original = {}
        self.run = ''

        tqdm.pandas()

    def process_documents(self, topics_path, doc_list_path, run):
        self.read_input(topics_path, doc_list_path, run)
        self.parse_and_index_docs()
        self.merge_index()

        self.inverted_index_dict_original = self.inverted_index_dict.copy()
        if self.doc_term_weighting == 'log':
            self.inverted_index_dict = self.log_weighting(
                self.inverted_index_dict)

    def read_input(self, topics_path, doc_list_path, run) -> None:
        with open(doc_list_path, "r") as f:
            self.doc_list = f.read().splitlines()
        if '_en' in doc_list_path:
            self.lang = 'en'
        else:
            self.lang = 'cs'

        self.selected_tags = self.tags[0][self.lang]
        self.doc_path = self.doc_path + self.lang + '/'

        self.doc_term_weighting = config[self.lang][run]['doc-term-weighting']
        self.doc_weighting = config[self.lang][run]['document-weighting']
        self.query_term_weighting = config[self.lang][run]['query-term-weighting']
        self.query_weighting = config[self.lang][run]['query-weighting']
        self.normalization = config[self.lang][run]['vector-normalization']
        self.run = run

        with open(topics_path, "r") as f:
            self.topics = f.read()

    def merge_index(self):
        print('Merge inverted indices')
        for doc_indices in tqdm(self.doc_index):
            dict_items = map(methodcaller('items'), doc_indices)
            for k, v in chain.from_iterable(dict_items):
                if k in self.inverted_index_dict.keys():
                    self.inverted_index_dict[k][list(
                        v.keys())[0]] = list(v.values())[0]
                else:
                    self.inverted_index_dict[k] = v

    def parse_and_index_docs(self):
        print('Parse and index documents')
        with Pool() as pool:
            self.doc_index = list(tqdm(
                pool.imap(self.parse_documents, self.doc_list)))

    def parse_documents(self, doc):
        full_doc_path = self.doc_path + doc
        inverted_index = []
        with open(full_doc_path, "r") as f:
            xml_doc = f.read()
            parsed_doc = BeautifulSoup(xml_doc, 'lxml')
            all_docs = parsed_doc.findAll('doc')
            for doc in all_docs:
                doc_id = doc.find('docid').text
                relevant_text = doc.findAll(self.is_tag_included)
                doc_content = ' '.join([relevant_text[i].text for i in range(
                    len(relevant_text)) if relevant_text[i].text != ''])
                doc_dict = {'id': doc_id, 'doc': doc_content, 'tokens': []}
                doc_dict['tokens'] = self.preprocess.preprocess(
                    doc_content, self.lang,self.run, lower_case=config[self.lang][self.run]['lowercase'], remove_stopwords=config[self.lang][self.run]['remove_stopwords'], stem=config[self.lang][self.run]['stemming'], lemmatize=config[self.lang][self.run]['lemmatize'])
                self.doc_length[doc_id] = len(doc_dict['tokens'])
                inverted_index.append(self.create_index(doc_dict))
                self.parsed_doc_list[doc_id] = doc_dict
        return inverted_index

    def is_tag_included(self, tag):
        return tag.name.upper() in self.selected_tags

    def create_index(self, doc):
        inverted_index = {}
        for token in doc['tokens']:
            if token in inverted_index.keys():
                freq = inverted_index[token][doc['id']
                                             ] if doc['id'] in inverted_index[token].keys() else 0
                inverted_index[token][doc['id']] = freq + 1
            else:
                inverted_index[token] = {doc['id']: 1}
        return inverted_index

    def create_query_index(self, tokens, topic_idx):
        for token in tokens:
            if token in self.term_topic_index.keys():
                freq = self.term_topic_index[token][topic_idx
                                                    ] if topic_idx in self.term_topic_index[token].keys() else 0
                self.term_topic_index[token][topic_idx] = freq + 1
            else:
                self.term_topic_index[token] = {topic_idx: 1}

    def create_queries(self):
        topic_list = re.findall(r'<num>(.+?)</num>', self.topics)
        self.topics = re.findall(r'<title>(.+?)</title>', self.topics)
        self.topic_title_map = [{
            'id': topic_list[i], 'query': self.topics[i], 'tokens': []} for i in range(len(topic_list))]
        print('Preprocess and index queries')
        for topic in tqdm(self.topic_title_map):
            topic['tokens'] = self.preprocess.preprocess(
                topic['query'], self.lang, self.run, lower_case=config[self.lang][self.run]['lowercase'], remove_stopwords=config[self.lang][self.run]['remove_stopwords'], stem=config[self.lang][self.run]['stemming'], lemmatize=config[self.lang][self.run]['lemmatize'])
            self.create_query_index(topic['tokens'], topic['id'])
        self.term_topic_index = self.log_weighting(self.term_topic_index)
        self.calculate_similarity()

    def calculate_idf(self, total_docs, doc_freq):
        return np.log10(total_docs/doc_freq)

    def log_weighting(self, inverted_index_dict):
        print('Log weighting')
        return {term: {doc: 1 + np.log10(term_freq) for doc, term_freq in posting_list.items()} for term, posting_list in inverted_index_dict.items()}

    def find_max_term_freq(self, doc,):
        doc_term_freq = [self.inverted_index_dict_original[token][doc]
                         for token in self.parsed_doc_list[doc]['tokens']]
        return max(doc_term_freq)

    def augmented_weighting(self, term_freq, max_term_freq):
        return 0.5 + (0.5*term_freq/max_term_freq)

    def prob_idf(self, total_docs, doc_freq):
        return max(0, np.log10((total_docs-doc_freq)/doc_freq))

    def calculate_doc_norm(self, doc):
        doc_unique_tokens = set(
            [token for token in self.parsed_doc_list[doc]['tokens']])
        return np.linalg.norm([self.inverted_index_dict[token][doc] for token in doc_unique_tokens])

    def find_pivot_value(self):
        doc_lengths = [len(self.parsed_doc_list[doc]['tokens'])
                       for doc in self.parsed_doc_list]
        return np.average(doc_lengths)

    def calculate_similarity(self):
        print("Calculate similarity score")
        for topic in tqdm(self.topic_title_map):
            similarities = {}
            for token in topic['tokens']:
                query_term_freq = self.term_topic_index[token][topic['id']]
                if self.query_weighting == 'idf':
                    query_term_freq *= self.calculate_idf(
                        len(self.topics), len(self.term_topic_index[token]))
                if token in self.inverted_index_dict.keys():
                    postings_list = self.inverted_index_dict[token]
                    for doc, freq in postings_list.items():
                        if self.doc_term_weighting == 'augmented':
                            freq = self.augmented_weighting(
                                freq, self.find_max_term_freq(doc))
                        if self.doc_weighting == 'idf':
                            freq *= self.calculate_idf(
                                len(self.parsed_doc_list), len(postings_list))
                        elif self.doc_weighting == 'prob-idf':
                            freq *= self.prob_idf(len(self.parsed_doc_list),
                                                  len(postings_list))
                        if doc not in similarities.keys():
                            similarities[doc] = 0
                        similarities[doc] += freq * query_term_freq
            if self.normalization is not None:
                pivot_value = self.find_pivot_value()
                for doc in similarities.keys():
                    doc_norm = self.calculate_doc_norm(doc)
                    if doc_norm > 0:
                        if self.normalization == 'cosine':
                            similarities[doc] /= (doc_norm)
                        elif self.normalization == 'pivot':
                            alpha = 0.75
                            norm_factor = alpha*doc_norm + \
                                (1-alpha)*pivot_value
                            similarities[doc] /= norm_factor
            self.final_ranking[topic['id']] = {k: v for k, v in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True)}

    def get_output(self, output_file, run):
        with open(output_file, "w") as f:
            for topic, doc_list in self.final_ranking.items():
                rank = 0
                for doc, score in doc_list.items():
                    if rank < 1000:
                        output_string = topic + '\t0\t' + doc + '\t' + \
                            str(rank) + '\t' + str(score) + '\t' + run + '\n'
                        f.write(output_string)
                        rank += 1
