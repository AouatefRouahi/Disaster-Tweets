import pandas as pd

# nlp librairies
import string
import re
import spacy

import demoji # a library to remove emojis

# global vars
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# we need to change the standard punctuation to make a special preprocessing with the '-' and '\''
punct = re.sub('[-\']', '',string.punctuation)

# Iterables method
from itertools import islice


# *****************************************************************************

def missing(dataset):
    missing = pd.DataFrame(columns=['Variable', 'n_missing', 'p_missing'])

    miss = dataset.isnull().sum() # series

    missing['Variable'] = miss.index
    missing['n_missing'] = miss.values
    missing['p_missing'] = round(100*miss/dataset.shape[0],2).values

    return missing.sort_values(by='n_missing')  

def exists(text, char):
    if text.find(char) != -1:
        print(text)

        
def extract_meta(sentence):
    # word_count
    word_count = len(str(sentence).split())

    # unique_word_count
    unique_word_count = len(set(str(sentence).split()))

    # stop_word_count
    stop_word_count = len([w for w in str(sentence).lower().split() if w in stopwords])

    # url_count
    url_count = len([w for w in str(sentence).lower().split() if 'http' in w or 'https' in w])

    # char_count
    char_count = len(str(sentence))

    # punctuation_count
    punctuation_count = len([c for c in str(sentence) if c in string.punctuation])

    # hashtag_count
    hashtag_count = len([c for c in str(sentence) if c == '#'])

    # mention_count
    at_count = len([c for c in str(sentence) if c == '@'])
    
    return word_count, unique_word_count, stop_word_count, url_count, char_count, punctuation_count, hashtag_count, at_count
    #return word_count

def valid(s):
    valid = False
    if len(s)>2:
        dup = True
        i=1
        while dup and i<len(s):
            if s[i] != s[i-1]:
                dup = False
            i+=1
        valid = not(dup)
    return valid


def preprocess_text(sentence):
    clean_tokens = ''
    
    # check if it is nan
    if sentence == sentence:
        # special processing for compound words such as long-term and possesive form such as shop's customers 
        sentence_w_hyphen = re.sub('[-\']',' ', sentence)
        
        # Remove cases (caseless matching like the german scharfes S : ß)
        sentence_w_caseless = "".join([i.casefold() for i in sentence_w_hyphen])
        
        # remove numbers
        sentence_w_num = ''.join(i for i in sentence_w_caseless if not i.isdigit())
        
        # remove links
        # https? --> http or https #\S+:1 or more characters which is not a whitespace 
        sentence_w_link = re.sub(r"https?://\S+", "", sentence_w_num)
        
        # remove @names
        sentence_w_at = re.sub(r"@\S+", "", sentence_w_link)
        
        # remove emojis
        sentence_w_emoji = demoji.replace(sentence_w_link, "")
        
        # lower and remove punctuation (!"#$%&\()*+,./:;<=>?@[\\]^_`{|}~)
        sentence_w_punct = "".join([i.lower() for i in sentence_w_emoji if i not in punct])
        
        #Remove extra spaces, tabs, and line breaks
        sentence_w_extra= " ".join(sentence_w_punct.split())
        
        # ******************************************************************************
        
        #tokenize, remove stopwords and invalid words (words contains the same caracter or with len <=2) 
        tokens = sentence_w_extra.split()
        tokens_w_stopwords = [token for token in tokens if token not in stopwords]
        valid_tokens = [token for token in tokens_w_stopwords if valid(token)]
        clean_tokens = ' '.join(valid_tokens)
        
        # ******************************************************************************
        # lemmatize 
        nlp_doc = nlp(clean_tokens)
        tokens_lemma = [token.lemma_ for token in nlp_doc]
        clean_tokens = ' '.join(token for token in tokens_lemma)
    
    return clean_tokens

def print_text(sample, clean):
    print(f"Before: {sample}")
    print(f"After: {clean}")
    
def sort_dict(dic):
    return dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))

def take(n, dic):
    '''
    Return first n items of a dictionary
    '''
    return dict(islice(dic, n))
    
def generate_list_ngrams(text, n_gram):
    tokens = text.split(' ')
    iter_ngrams = zip(*[tokens[i:] for i in range(n_gram)])           # Iterator over list(tuples(length=n_gram))
    return [' '.join(ngram) for ngram in iter_ngrams]                 # list (strings)  

def generate_ngrams(texts, n_gram):
    n_grams = dict()
    for text in texts:
        for gram in generate_list_ngrams(text, n_gram):
            try:
                n_grams[gram] += 1
            except KeyError:
                n_grams[gram] = 1 
            '''
            if gram not in n_grams.keys():
                n_grams[gram] = 1
            else:
                n_grams[gram] += 1
            '''              
    return sort_dict(n_grams)