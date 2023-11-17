import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer


######################################
# Function for text pre-processing   #
######################################
# Input: Text
# Output: List with words
def preprocess(text_data,
               lower=True,
               rem_punc=True,
               word_tokenization=True,
               rem_numb=True,
               rem_stopwords=True,
               extra_stopwords=[],
               stem=True,
               lem=True):
    # extra_stopwords = ["n\'t","’","\'s"]
    tokens = text_data
    if lower:
        #lowercase text
        tokens = text_data.lower()

    if word_tokenization:
        # tokenize text by words
        tokens = word_tokenize(tokens)

    if rem_numb:
        # # Remove numbers
        tokens = [w for w in tokens if not w.isdigit()]

    if rem_punc:
        # # remove punctuation
        tokens = [token for token in tokens if not token in string.punctuation]
    if rem_stopwords:
        # # remove stopwords
        lang_stopwords = stopwords.words("english")
        lang_stopwords += extra_stopwords
        tokens = [token for token in tokens if token not in lang_stopwords]
    if stem:
        # # stemming
        lang = "english"
        stemmer = SnowballStemmer(lang)
        tokens = [stemmer.stem(token) for token in tokens]
    if lem:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens
