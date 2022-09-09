import os
from tkinter import filedialog
from tkinter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
import numpy as np

# set the root path to the current python script location
abspath = os.path.abspath(__file__)
root = os.path.dirname(abspath)
os.chdir(root)

# open a new window to select the folder containing the text docs
# this function is finished
def get_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    folder_path = "documents/"
    return folder_path


# build the doc name - text string dictionary
def get_docDict(path):
    doc_dict = {}
    file_names = os.listdir(path)

    for file in file_names:
        full_path = path+'/'+file
        with open(full_path, 'r', errors='ignore') as f:
            data = f.readlines()
        text = "".join([i for i in data])
        # remove all the "\n" from the text
        text = re.sub("\n", " ", text)
        doc_dict[file] = text
    return doc_dict


#  clean the text by removing the unnecessary characters and split into tokens
def clean_text(doc_dict):
    """
    input - a dictionary of {filename : text}
    output - a dictionary of {filename : clean text} 

    """
    clean_dict = {}
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    for name, doc in doc_dict.items():
        # remove extra white space
        text = re.sub(r"\s+", " ", doc)
        # remove extra ...
        text = re.sub(r"\.+"," ", doc)
        # remove hyphen
        text = re.sub(r"-","", text)
        text = text.lower()
        text_tokens = word_tokenize(text)
        text_clean = []
        for word in text_tokens:
            if (word not in stopwords_english and word not in string.punctuation):
                stem_word = stemmer.stem(word)
                text_clean.append(word)
        
        clean_dict[name] = text_clean
        
    return clean_dict





# make the vocabulary of the doc collection
def make_vocab(doc_dict):
    total_tokens = []
    for tokens in doc_dict.values():
        total_tokens += tokens
    vocab = list(set(total_tokens))
    return vocab

# get the Term Frequency table
def get_DocTF(doc_dict, vocab):
    """
    input - a dictionary of {filename : clean text}, the vocabulary of the whole dataset
    output - a dictionary of {filename : {term : count}}
    """
    tf_dict = {}
    # make the dict for filename=>{term:frequency}
    for doc_id in doc_dict.keys():
        tf_dict[doc_id] = {}

    for word in vocab:
        for doc_id, text in doc_dict.items():
            tf_dict[doc_id][word] = text.count(word)
        
    return tf_dict


# get the doc frequency table
def get_DocDF(clean_dict, vocab):
    """
    input - a dictionary of {filename : clean text}, the vocabulary of the whole dataset
    output - a dictionary of all terms in the vocabulary - {term : count}
    """
    df_dict = {}
    for word in vocab:
        freq = 0
        for text_tokens in clean_dict.values():
            if word in text_tokens:
                freq += 1
        df_dict[word] = freq

    return df_dict


# get the inverse doc frequency table
def inverse_DF(df_dict, vocab, doc_length):
    """
    input - a dictionary of DF {term : count}, the vocabulary of the whole dataset, total # of documents in the dataset
    output - a dictionary of IDF of all terms in the vocabulary - {term : inver_df}
    """
    idf_dict = {}
    for word in vocab:
        # idf_dict[word] = - np.log2((df_dict[word]) / (doc_length)) 
        idf_dict[word] = round(np.log(((doc_length - df_dict[word]+0.5) / (df_dict[word]+0.5))+1), 4)
        
    return idf_dict

# calculate the TF-IDF table
def get_tf_idf(tf_dict, idf_dict, doc_dict, vocab):
    tf_idf_dict = {}
    for doc_id in doc_dict.keys():
        tf_idf_dict[doc_id] = {}
    
    for word in vocab:
        for doc_id, text_tokens in doc_dict.items():
            tf_idf_dict[doc_id][word] = round((tf_dict[doc_id][word] * idf_dict[word]), 4)
    return tf_idf_dict


# the VSM ranking function - return the top-5
def vectorSpaceModel(query, doc_dict,tf_idf_dict):
    query_vocab = []
    query = query.lower()
    query = re.sub(r"\s+", " ", query)
    stopwords_english = stopwords.words('english')

    for word in query.split():
        if (word not in string.punctuation and word not in stopwords_english):
            query_vocab.append(word)

    query_wc = {}
    for word in query_vocab:
        query_wc[word] = query.split().count(word)


    relevance_scores = {}
    for doc_id in doc_dict.keys():
        score = 0
        for word in query_vocab:
            score += query_wc[word] * tf_idf_dict[doc_id][word]
        relevance_scores[doc_id] = round(score,4)

    # sort the relevance score and get the top-k ranking
    # sort the keys of the relevance score by value
    sort_keys = sorted(relevance_scores, key=relevance_scores.get , reverse = True)
    top_keys = sort_keys[:5]
    top_5 = {}
    for key in top_keys:
        top_5[key] = relevance_scores[key]

    return top_5

# calculate average document length
def get_avgdl(clean_dict):
    total_doc = len(clean_dict.keys())
    total_length = 0
    for text in clean_dict.values():
        total_length += len(text)

    avgdl = total_length / total_doc

    return round(avgdl, 4)


# calculate the BM25 term table
def bm25(tf_dict, clean_dict, df_dict, vocab, k=1.2, b=0.75):
    bm25_dict = {}
    avgdl = get_avgdl(clean_dict)
    N = len(clean_dict.keys())
    
    # create the collection of dictionaries of all documents
    for doc_id in clean_dict.keys():
        bm25_dict[doc_id] = {}

    for word in vocab:
        for doc_id, text_tokens in clean_dict.items():
            freq = tf_dict[doc_id][word]
            # the TF in BM25
            tf = (freq*(k+1)) / (freq + k*(1-b+b*len(clean_dict[doc_id])/avgdl))
            # get DF
            N_q = df_dict[word]
            idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
            score = round(tf*idf,4)
            bm25_dict[doc_id][word] = score

    return bm25_dict


# the BM25 ranking function - return the top-5
def BM25Model(query, doc_dict, bm25_dict):
    query_vocab = []
    query = query.lower()
    query = re.sub(r"\s+", " ", query)
    stopwords_english = stopwords.words('english')

    for word in query.split():
        if (word not in string.punctuation and word not in stopwords_english):
            query_vocab.append(word)

    query_wc = {}
    for word in query_vocab:
        query_wc[word] = query.split().count(word)

    

    relevance_scores = {}
    # use the raw doc_dict to get the filename only
    for doc_id in doc_dict.keys():
        score = 0
        for word in query_vocab:
            score += query_wc[word] * bm25_dict[doc_id][word]
        relevance_scores[doc_id] = round(score,4)

    # sort the relevance score and get the top-k ranking
    # sort the keys of the relevance score by value
    sort_keys = sorted(relevance_scores, key=relevance_scores.get , reverse = True)
    top_keys = sort_keys[:5]
    top_5 = {}
    for key in top_keys:
        top_5[key] = relevance_scores[key]

    return top_5