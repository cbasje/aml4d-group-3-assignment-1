import matplotlib.pyplot as plt
from wordcloud import WordCloud


def wordcloud(words,name_of_output,num):
  # you put in the function the name of the object
  # the function clones it under consistent name
  tokens = words
  # you put in the function the name of the object
  # the function clones it under consistent name
  name = str(name_of_output)
  
  # you pass the list of lists (each element is a word)
  # this line transforms it into one large text
  text = " ".join([" ".join([str(item) for item in sublist]) for sublist in tokens])

  # wordcloud settings, the key parameter here is max_words -- we set it in the function
  wordcloud = WordCloud(width = 1600, height = 800, max_words = num, background_color = 'white').generate(text)
  plt.imshow(wordcloud, interpolation = 'bilinear')
  plt.axis("off")
  plt.savefig(name + '.png') # if you want to save the WordCloud


def find_by_word(list_of_tokens, word_to_find):
  # you put in the function the name of the object
  # the function clones it under consistent name
  tokens = list_of_tokens

  # you pass the list of lists (each element is a word)
  # this line transforms it into a list of texts
  res = [' '.join(ele) for ele in tokens]
  # filter the object by the word / phrase you set in the function
  filter_object = filter(lambda a: word_to_find in a, res)
 
  result = list(filter_object)
  return result
