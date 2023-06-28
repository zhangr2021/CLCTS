import stanza
import spacy_stanza
import textdescriptives as td
import numpy as np
import pandas as pd
import spacy
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.system("export CUDA_VISIBLE_DEVICES=\"\"")
from multiprocessing import Pool, freeze_support
import sys
#stanza.download("en")
#stanza.download("de")
spacy.require_cpu()
nlp_en = spacy_stanza.load_pipeline('en', disable=['ner'])
nlp_en.add_pipe("textdescriptives/dependency_distance")
spacy.require_cpu()

def tree_height(root):
    """
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def get_average_heights(paragraph, nlp):
    """
    Computes average height of parse trees for each sentence in paragraph.
    :param paragraph: spacy doc object or str
    :return: float
    """
    if type(paragraph) == str:
        spacy.require_cpu()
        doc = nlp(paragraph)
    else:
        doc = paragraph
    roots = [sent.root for sent in doc.sents]
    # all attributes are stored as a dict in the ._.dependency_distance attribute
    dct = doc._.dependency_distance
    dct["mean_height"] = np.mean([tree_height(root) for root in roots])
    return pd.DataFrame(dct, index = [0])


def compute_score(row, nlp = nlp_en):
    idx, row = row[0]
    if type(row) == str:
        text  = row
    else:
        text = row["sentence"]
    text = (u"" + text)
    scoredf = get_average_heights(text, nlp)
    scoredf["sentence"] = text
    return scoredf

english = pd.read_pickle("./statistics_english_sent.pkl").iloc[:10]
print(len(english))

agg = zip(english.iterrows(), [nlp_en])
def main():
    with Pool() as pool:
        L = pool.starmap(compute_score, agg)
    return L

if __name__=="__main__":
    freeze_support()
    L = main()
    res = pd.DataFrame()
    for df in L:
        res = res.append(df)
    res.to_csv("./parsing/english_parsing_result_stanford_all.csv", index=False)   
               
