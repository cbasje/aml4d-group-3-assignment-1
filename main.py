from preprocessing import preprocess
from sentiment_analysis import calculate_sentiment, find_by_sentiment
from wordclouds import wordcloud, find_by_word
import nltk
import os
import pandas as pd
import tempfile
from topic_modelling import lda_topic_model, show_topics, show_example_sentences_by_topic, show_examples_for_all_topics
from gensim.models.ldamulticore import LdaMulticore

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
pd.set_option('display.max_columns', None)

# This line reads the file named "students_data.csv" in folder "data".
students_data = pd.read_csv("data/students_data.csv")

# This removes the empty rows
students_data = students_data['Needs'].loc[students_data['Needs'].notna()]

# In order to analyse the data with Voyant we export the Needs column to CSV
# df = pd.DataFrame(students_data, columns=['Needs'])
# df.to_csv("data/voyant_needs_export.csv")

# We merge all the different lines responses into one big corpus
corpus = students_data.to_list()

# Outputting the first 5 lines of the corpus to check the information
# print(corpus[0:5])
# print("\n")

tokens = [
    preprocess(sentence,
               lower=True,
               word_tokenization=True,
               rem_punc=True,
               rem_numb=True,
               rem_stopwords=True,
               extra_stopwords=[
                   "study", "studies", "student", "students", "feel",
                   "feelings"
               ],
               stem=True,
               lem=True) for sentence in corpus
]

# Outputting the first 5 lines of the tokens to check the tokenised information
# print(tokens[0:5])
# print("\n")

#############################################
#             WORD FREQUENCIES              #
#############################################

# This line puts the words in a wordcloud
# wordcloud(words=tokens, name_of_output='wordcloud', num=50)

# In order to understand the context of words, we get the whole sentences
token_in_context = find_by_word(tokens, 'need')
print(token_in_context)

#############################################
#            Sentiment analysis             #
#############################################

# Use only sentences with 5 or more tokens
min_len = 5

sent_result = calculate_sentiment(tokens,corpus, min_len=min_len)
# print(sent_result)

positive_sentiment = find_by_sentiment(df_with_scores=sent_result,
                                       score_type='pos',
                                       num_of_examples=5)
print(positive_sentiment)
compound_sentiment = find_by_sentiment(df_with_scores=sent_result,
                                       score_type='compound',
                                       num_of_examples=5)
print(compound_sentiment)
negative_sentiment = find_by_sentiment(df_with_scores=sent_result,
                                       score_type='neg',
                                       num_of_examples=5)
print(negative_sentiment)

#############################################
#         TOPIC MODELING                    #
#############################################

num_of_topics = 5

# lda_model = lda_topic_model(tokens,
#                             corpus,
#                             topic_num=num_of_topics,
                            # min_len=min_len)
lda_model = LdaMulticore.load("lda_model")

word_num_per_topic = 7
# show_topics(lda_model, word_num_per_topic)

num_of_examples = 5

# show_examples_for_all_topics(corpus,
#                              lda_model,
#                              tokens,
#                              word_num_per_topic=word_num_per_topic,
#                              num_of_examp_to_show=num_of_examples,
#                              min_len=min_len)

topic_id = 1
# show_example_sentences_by_topic(corpus,
#                                 tokens,
#                                 lda_model,
#                                 word_num_per_topic,
#                                 topic_to_check=topic_id,
#                                 num_of_examp_to_show=num_of_examples,
#                                 min_len=min_len)