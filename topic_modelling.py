from gensim.models import LdaModel
import gensim
import gensim.corpora as corpora
import numpy as np
from pprint import pprint
import pandas as pd
import pyLDAvis


########################
#    TOPIC MODELING.   #
########################
def lda_topic_model(tokens, corpus, topic_num, min_len=None):
    if min_len != None:
        corpus = [
            corpus[i] for i in range(len(corpus)) if len(tokens[i]) >= min_len
        ]
        tokens = [lst for lst in tokens if len(lst) >= min_len]

    # Create Dictionary
    id2word = corpora.Dictionary(tokens)
    # Create Corpus
    texts = tokens
    # Term Document Frequency
    corpus_2 = [id2word.doc2bow(text) for text in texts]
    # number of topics
    num_topics = topic_num
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus_2,
                                           id2word=id2word,
                                           num_topics=num_topics)

    lda_model.save("lda_model")
    return lda_model


def show_topics(lda_model, word_num_per_topic):
    for topic in lda_model.print_topics(num_words=word_num_per_topic):
        print("[Topic " + str(topic[0]) + "]: " + topic[1])


def filter_corpus_by_topic(corpus,
                           tokens,
                           lda_model,
                           topic_index_wanted,
                           min_len=None):
    if min_len != None:
        corpus = [
            corpus[i] for i in range(len(corpus)) if len(tokens[i]) >= min_len
        ]
        tokens = [lst for lst in tokens if len(lst) >= min_len]

    id2word = corpora.Dictionary(tokens)
    corpus_2 = [id2word.doc2bow(text) for text in tokens]
    topic_probabilities = lda_model[corpus_2]
    topic_predictions = []
    for p in topic_probabilities:
        prediction = p[np.argmax(p, axis=0)[1]][0]
        topic_predictions.append(prediction)
    topic_predictions = np.array(topic_predictions)
    corpus = np.array(corpus)
    return corpus[topic_predictions == topic_index_wanted]


def show_example_sentences_by_topic(corpus,
                                    tokens,
                                    lda_model,
                                    word_num_per_topic,
                                    topic_to_check,
                                    num_of_examp_to_show,
                                    min_len=None):
    if min_len != None:
        corpus = [
            corpus[i] for i in range(len(corpus)) if len(tokens[i]) >= min_len
        ]
        tokens = [lst for lst in tokens if len(lst) >= min_len]

    print(
        "\n\n######################\n# TOPIC and Examples #\n######################\n"
    )
    print("[Topic " + str(topic_to_check) + "]: " + lda_model.print_topics(
        num_words=word_num_per_topic)[topic_to_check][1])

    filtered_corpus = filter_corpus_by_topic(corpus, tokens, lda_model,
                                             topic_to_check, min_len)
    for s in filtered_corpus[0:num_of_examp_to_show]:
        print("\n")
        print(s)


def show_examples_for_all_topics(corpus,
                                 lda_model,
                                 tokens,
                                 word_num_per_topic,
                                 num_of_examp_to_show,
                                 min_len=None):
    if min_len != None:
        corpus = [
            corpus[i] for i in range(len(corpus)) if len(tokens[i]) >= min_len
        ]
        tokens = [lst for lst in tokens if len(lst) >= min_len]

    data = []

    topics = lda_model.get_topics()
    num_unique_topics = len(topics)

    for topic_id, topic_words in lda_model.show_topics(
            num_words=word_num_per_topic,
            formatted=False,
            num_topics=num_unique_topics):
        filtered_corpus = filter_corpus_by_topic(corpus, tokens, lda_model,
                                                 topic_id, min_len)
        examples = filtered_corpus[0:num_of_examp_to_show]
        #example = ', '.join([str(s) for s in example])
        for example in examples:
            data.append(
                (topic_id, ', '.join([word
                                      for word, _ in topic_words]), example))

    df = pd.DataFrame(data, columns=['Topic ID', 'Topic Words', 'Example'])

    df.to_csv("examples_of_topics.csv")
