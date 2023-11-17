# import necessary documents
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def calculate_sentiment(list_of_tokens, corpus, min_len=None):
  tokens = list_of_tokens

  if min_len != None:
    corpus = [corpus[i] for i in range(len(corpus)) if len(tokens[i]) >= min_len]
    tokens = [lst for lst in tokens if len(lst) >= min_len]

  # you put in the function the name of the object
  # the function clones it under consistent name

  # you pass the list of lists (each element is a word)
  # this line transforms it into a list of texts
  a_list = [' '.join(ele) for ele in tokens]

  # this lines takes each text from the collection, calculated scores, and then merge calculated scores
  scores = []
  analyzer = SentimentIntensityAnalyzer()
  for sentence in a_list:
    vs = analyzer.polarity_scores(sentence)
    scores.append(vs)

  # after scores are calculated, for presentation purposes the code tranforms lists into a dataframe and merge sentences with calculated scores
  data = pd.DataFrame(a_list, columns=['sentences'])
  data2 = pd.DataFrame(scores)
  data3 = pd.DataFrame(corpus)
  final_dataset = pd.concat([data, data2, data3], axis=1)
  return final_dataset


def find_by_sentiment(df_with_scores, score_type, num_of_examples):
  # you put in the function the name of the object
  # the function clones it under consistent name
  final_dataset = df_with_scores

  # sort resulting dataframe based on the score type you're interested in
  final_dataset = final_dataset.sort_values(by=[score_type], ascending=False)
  # show top N of examples
  res = final_dataset.head(num_of_examples)
  return res
