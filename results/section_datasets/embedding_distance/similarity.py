import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from multiprocessing import Pool, freeze_support
import sys

dataset_path, language = sys.argv[-2:]

class embedding_dist():
    def __init__(self, path_embedding, chunksize):
        self.path_embedding = path_embedding
        self.chunksize = chunksize
    
    def retrieve_data(self, chunk, doc_col, sum_col):
        cumm = pd.DataFrame()
        for df in chunk:
            tmp = df[(df.dataset == self.dataset) & (df["col"]  == doc_col)]
            cumm = pd.concat([cumm,tmp])
            print(sum_col)
            tmp = df[(df.dataset == self.dataset) & (df["col"] == sum_col)]
            cumm = pd.concat([cumm,tmp])
            if (len(tmp) ==0) & (len(cumm) >0):
                break
            #print(len(tmp), len(cumm))
            #if len(cumm)> 10000:
             #   break
        self.cumm = cumm
        return self.cumm

    # load huge dataset
    def load_data(self, doc_col, sum_col):
        df = pd.DataFrame()
        if self.language == "mul":
            language = "embedding"
        else:
            language = self.language
        for file in os.listdir(self.path_embedding):
            if ("_sent" in file) & (language in file):
                chunk = pd.read_csv(self.path_embedding + file, chunksize = self.chunksize)
                tmp = self.retrieve_data(chunk,  doc_col, sum_col)
                tmp["direction"] = file[:2]
                df = df.append(tmp)
            #if len(df) > 50000:
             #   break
        print(self.dataset, df.shape[0], df.col.unique())
        print(df.head())
        return df

    def similarity(self, doc_, sum_, start, end):
        doc_similarity = []
        for idx, row in sum_.iterrows():
            sum_emb = row[start:end]
            for idx_, row_ in doc_.iterrows():
                doc_emb = row_[start:end]
                doc_similarity.append(cosine_similarity(np.array(sum_emb).reshape(1, -1), 
                                          np.array(doc_emb).reshape(1, -1))[0][0])
                #print(np.array(sum_emb).reshape(1, -1))
        return np.mean(doc_similarity)
    
    def return_distance(self, dataset, doc_col, sum_col, language):
        self.dataset = dataset
        self.language = language
        df = self.load_data(doc_col, sum_col)
        start = df.columns.tolist().index("0")
        end = df.columns.tolist().index("767")
        doc_sim = {}
        for id_ in df.id.unique():
            doc_ = df[(df.id == id_) & (df.col == doc_col)]
            sum_ = df[(df.id == id_) & (df.col == sum_col)]
                #print(doc_ge, sum_ge)
            if (len(doc_)>0) & (len(sum_)>0):
                doc_sim[str(id_)]= self.similarity(doc_, sum_, start = start, end = end)
        similarity_doc = pd.DataFrame(doc_sim, index = [0]).T.reset_index()
        similarity_doc.columns = ["id", "similarity"]
        similarity_doc["dataset"] = self.dataset
        similarity_doc["doc-sum"] = "-".join([doc_col, sum_col])
        return similarity_doc
# sum0 german col, sum1 english col

embedding = embedding_dist(dataset_path,  chunksize = 10000)

#CLCT_hDe          [reference_summary_tl]
#CLCT_hEn    [text, reference_summary_sl]
#cnndm              [article, highlights]
#wiki           [summary_en, document_en]
#CLCT_hDe    [text, reference_summary_sl]
#CLCT_hEn          [reference_summary_tl]
#histsum                 [story, summary]
#mlsum                     [doc, summary]
#wiki           [summary_de, document_de]


agg = [("histsum", "story", "summary", "german"), 
       ("CLCT_hDe", "text", "reference_summary_sl", "german"),
       ("CLCT_hDe", "text", "reference_summary_tl", "mul"), 
       ("cnndm", "article", "highlights", "english"), 
       ("CLCT_hEn", "text", "reference_summary_tl", "mul"),
       ("CLCT_hEn", "text", "reference_summary_sl", "english"),
       ("wiki", "document_en", "summary_de", "mul"),
       ("wiki", "document_en", "summary_en", "english"),
       ("wiki", "document_de", "summary_en", "mul"), 
      ("wiki", "document_de", "summary_de", "german"),
      ("mlsum", "doc", "summary", "german")]
def main():
    with Pool() as pool:
        L = pool.starmap(embedding.return_distance, agg)
    return L

if __name__=="__main__":
    freeze_support()
    L = main()
    res = pd.DataFrame()
    for df in L:
        res = res.append(df)
    res.to_csv("./dataset_statistics/all_datasets_embedding_distance.csv")   
               