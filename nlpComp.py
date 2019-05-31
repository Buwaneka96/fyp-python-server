from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, state_union
from nltk import pos_tag
import spacy

def get_elements(data):
    # data = "User must enter both email and password"

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(data)

    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    print(word_tokens)
    print(filtered_sentence)

    print('\n')
    print(data)
    tagged = pos_tag(word_tokens)
    print(tagged)

    elements = [];

    for (word, tag) in tagged :
        if tag in ('NN', 'NNP'):
            print(word)
            elements.append(word)

    print('elements : ',elements)
    return elements


# nlp = spacy.load('en_core_web_sm')
# apples, and_, oranges = nlp(u'mail and email')
# print(apples.similarity(oranges))