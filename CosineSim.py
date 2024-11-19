import random
import statistics
from collections import Counter

import it
from matplotlib import pyplot, pyplot as plt
import nltk
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize
import sys
#import gensim.downloader as api
import numpy as np
import gensim as api

from gensim.gensim.parsing import remove_stopwords

from gensim.gensim import parsing

class CosineSim:
    sys.path.append('/gensim')  # Add the folder containing gensim

    @staticmethod
    def random_wiki_articles():
        num_of_articles = 500
        generate_random_articles = wikipedia.random(num_of_articles)
        retrieve_titles = []
        retrieve_contents = []
        page_ids = []

        for articles in generate_random_articles:
            try:
                wikipedia.set_lang('en')
                wikipedia_page_object = wikipedia.WikipediaPage(title=articles,
                                                        pageid=True,
                                                        redirect=True,
                                                        preload=False,
                                                        original_title=u'')
                lowercase_titles = str(wikipedia_page_object.title).lower()
                lowercase_contents = str(wikipedia_page_object.html()).lower()
                cleantext = BeautifulSoup(lowercase_contents, "html").text
                page_ids.append(wikipedia_page_object.pageid)
                retrieve_contents.append(cleantext)
                retrieve_titles.append(lowercase_titles)
            except wikipedia.exceptions.DisambiguationError as e:
                e.options
            except wikipedia.exceptions.PageError as e:
                print(page_ids)
        return retrieve_titles, retrieve_contents

    @staticmethod
    def preprocessing():
        titles_list, contents_list = CosineSim.random_wiki_articles()
        removes_stop_words_titles = []
        rv_st_content = []
        num_of_words = 25
        pre_processing_titles = dict()
        pre_processing_contents = dict()
        pre_processing_contents_random = dict()
        for titles in titles_list:
            titles = remove_stopwords(titles)
        removes_stop_words_titles.append(titles)
        for index in range(len(removes_stop_words_titles)):
            tokenized_titles = word_tokenize(removes_stop_words_titles[index])
        tokenized_titles = [word.lower() for word in tokenized_titles if
                            word.isalpha()]
        title_dictionary = dict.fromkeys(tokenized_titles, index)
        pre_processing_titles[index] = list(title_dictionary.keys())
        for content in contents_list:
            content = remove_stopwords(content)
        rv_st_content.append(content)
        for jindex in range(len(rv_st_content)):
            tokenized_contents = word_tokenize(rv_st_content[jindex])
        tokenized_contents = [word.lower() for word in tokenized_contents
                              if word.isalpha()]
        random.shuffle(tokenized_contents)
        content_dictionary_frequent = dict(Counter(tokenized_contents).most_common(num_of_words))
        n = len(Counter(tokenized_contents))
        content_dictionary_random = tokenized_contents[:num_of_words] if n > num_of_words else tokenized_contents
        pre_processing_contents[jindex] = list(content_dictionary_frequent.keys())
        pre_processing_contents_random[jindex] = list(content_dictionary_random)
        tuple_frequents = tuple(
            zip(list(pre_processing_titles.values()),
                list(pre_processing_contents.values())))

        tuple_randoms = tuple(
            zip(list(pre_processing_titles.values()),
                list(pre_processing_contents_random.values())))
        return tuple_frequents, tuple_randoms

    @staticmethod
    def get_similarity_score():
        tuple_frequents, tuple_randoms = CosineSim.preprocessing()
        model = api.load('word2vec-google-news-300')
        cosine_sim_frequent_dictionary = {}
        cosine_sim_random_dictionary = {}
        values_not_present_frequent = []
        values_not_present_random = []
        total_values_frequent = []
        total_values_random = []
        for frequent_k, frequent_v in tuple_frequents:
            for fkey, fvalue in it.product(frequent_k, frequent_v):
                total_values_frequent.append(fvalue)
            try:
                cosine_sim_frequent_dictionary[fvalue] = model.similarity(fkey, fvalue)
            except KeyError:
                values_not_present_frequent.append(fvalue)
            for random_k, random_v in tuple_randoms:
                for rkey, rvalue in it.product(random_k, random_v):
                    total_values_random.append(rvalue)
            try:
                cosine_sim_random_dictionary[rvalue] = model.similarity(rkey, rvalue)
            except KeyError:
                values_not_present_random.append(rvalue)
            percentage_frequency = (len(values_not_present_frequent) /
                                    len(total_values_frequent)) * 100
            print(f"Percentage of words not present for frequent words is {percentage_frequency} % ")
            percentage_random = (len(values_not_present_random) /
                                 len(total_values_random)) * 100
            print(f"Percentage of words not present for random words is {percentage_random} % ")
            return cosine_sim_frequent_dictionary, cosine_sim_random_dictionary


def histogram_plot():
    cosine_sim_frequent_dictionary, cosine_sim_random_dictionary = CosineSim.get_similarity_score()
    try:
        x_value_frequent = list(cosine_sim_frequent_dictionary.values())

        x_value_random = list(cosine_sim_random_dictionary.values())
        x_values = [x_value_frequent, x_value_random]
        labels = ['frequent words', 'random words']
    except statistics.StatisticsError:
        print("Data is empty")
        bins = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plt.hist(x_values, bins=bins, label=labels, alpha=0.5,
                 edgecolor='black')
        plt.legend(loc='best')
        plt.title('Similarity Score: Frequent words vs Random words')
        plt.ylabel('frequency')
        plt.xlabel('cosine scores')
        plt.grid(axis='y')
        plt.show()


alg = CosineSim()
alg.histogram_plot()