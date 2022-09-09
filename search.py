import get_index

# return the formatted top-5 docs by VSM ranking
def rank_by_VSM(query, doc_dict, tfidf_dict):
    # implement here
    # get the top 5 docs
    
    result = get_index.vectorSpaceModel(query, doc_dict, tfidf_dict)
    outputs = "Using VSM to rank...\n"

    # format the outputs
    for key in result:
        outputs += "Doc ID: " + str(key) + " with score: " + str(result[key]) + "\n"
    

    return outputs

# return the formatted top-5 docs by BM25 ranking
def rank_by_BM25(query, doc_dict, tf_dict, clean_dict, df_dict, vocab):
    # implement here
    # get the top 5 docs
    bm25_dict = get_index.bm25(tf_dict, clean_dict, df_dict, vocab)
    result = get_index.vectorSpaceModel(query, doc_dict, bm25_dict)

    outputs = "Using BM25 to rank...\n"

    # format the outputs
    
    for key in result:
        outputs += "Doc ID: " + str(key) + " with score: " + str(result[key]) + "\n"

    return outputs





