import pandas as pd
import os
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score 
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# load data
output_all_hEn = pd.read_csv("./eval_metric_hEn.csv")
merged_hEn = pd.read_csv("./hEn-human_chatGPT_annotation_v2.csv")
print("shape of all outputs:", output_all_hEn.shape[0], "\n shape of all human annotation:",merged_hEn.shape[0])

order_metric = ['rouge1', 'rougel', 'bertscore_P',
       'bertscore_R', 'bertscore_F1', 
                'bartscore', 'moverscore', 'menli','MENLI_W0.8', 'MENLI_W0.3', 'MENLI_W0.2', "DiscoScore_F", "DiscoScore_S"]

columns_list = ['coherence', 'consistency', 'fluency', 'relevance', 'rouge1', 'rougel', 'bartscore', 'moverscore',
       'bertscore_P', 'bertscore_R', 'bertscore_F1', 'menli', 'MENLI_W0.8',
       'MENLI_W0.3','MENLI_W0.2', "DiscoScore_F", "DiscoScore_S"]

model_order = ['English_25_False', 'English_25_True', 'English_100', "1", "2", "3", "4", "5", "6", "11", 
               'chatGPT_title', "chatGPT_e2e",  "chatGPT_pipeline"]


# generate results table 1 eval_metrics
filtered = output_all_hEn.set_index("id").loc[output_all_hEn[output_all_hEn.model_id == "English_100"].id]
latex = filtered.groupby("model_id").mean().iloc[:, -14:].loc[model_order, order_metric].round(4)

def turn_string(row):
    lst = ["{:.3f}".format(i.round(3)) for i in row.values]
    lst_ = ["/".join(lst[:2]), "/".join(lst[2:5]), lst[5], lst[6], "/".join(lst[7:-2]), lst[-2]]
    print(lst_)
    return "&".join(lst_)

#generate latex
pd.DataFrame(latex.apply(lambda x: turn_string(x), axis = 1) + "\\" + "\\").to_csv("./outputs/hEn_latex_experiment_multilingual.csv", index = True)

# human-chatty annotation
merged_hEn["id_model"] = merged_hEn.apply(lambda x: str(x["id"]) + "_" + x["model_id"], axis = 1)
intersection_of_models = set(merged_hEn[(merged_hEn.annotator !=6)].id_model.unique()).intersection(set(merged_hEn[(merged_hEn.annotator == 6)].id_model.unique()))

# human annotation result avg. 
print("human annotaiton All")
summary_level = pd.DataFrame(merged_hEn[(merged_hEn.annotator != 6)].set_index("id_model").loc[intersection_of_models].groupby(["model_id", "id"]).mean()).reset_index()
human_mean = summary_level.groupby("model_id").mean().loc[model_order].iloc[:, 1:5]

# average ChatGPT annotation results
summary_level = pd.DataFrame(merged_hEn[merged_hEn.annotator == 6].set_index("id_model").loc[intersection_of_models].groupby(["model_id", "id"]).mean()).reset_index() 
chatGPT_mean = summary_level.groupby("model_id").mean().loc[model_order].iloc[:, 1:5]

order_anno =["coherence_human","coherence_chatGPT", "consistency_human", "consistency_chatGPT",
             "fluency_human", "fluency_chatGPT", "relevance_human",   'relevance_chatGPT']
annotation_human_chatty = pd.merge(human_mean.reset_index(), chatGPT_mean.reset_index(), on = "model_id", 
         suffixes = ["_human", "_chatGPT"], how = "outer").set_index("model_id").loc[model_order, order_anno].round(2)

def turn_string(row):
    lst = ["{:.2f}".format(i.round(3)) for i in row.values]
    lst_ = ["/".join(lst[:2]), "/".join(lst[2:4]), "/".join(lst[4:6]), "/".join(lst[6:])]
    print(lst_)
    return "&".join(lst_)
#generate latex
pd.DataFrame(annotation_human_chatty.apply(lambda x: turn_string(x) + "\\" + "\\", axis = 1)).to_csv("./outputs/hEn_latex_human_chatty_anno.csv", index = True)

# annotation aggrement
def agreement_return(dim, annotator_lst_hEn, merged_hEnidlst, df, exclude = True):
    tmp = agreement(dim, annotator_lst_hEn, merged_hEnidlst, df)
    cor_lst = []
    for a1 in tmp.index:
        if exclude:
            if dim == "fluency":
                a1 = int(a1)+1 # exclude fluency from annotator 1
        for a2 in  tmp.index:
            if int(a2)>int(a1):
                agree = tmp.loc[[a1,a2]].dropna(axis='columns')
                if len(agree.loc[a1]) > 0:
                    cor = spearmanr(agree.loc[a1], agree.loc[a2]).correlation
                    k1 = [int(val) for val in agree.loc[a1]]
                    k2 = [int(val) for val in agree.loc[a2]]
                    kappa = cohen_kappa_score(k1, k2)
                    cor_lst.append((a1, a2, cor, kappa, len(agree.loc[a1]), agree.columns.tolist())) 
                else: print("length problem", agree)
    return cor_lst

def agreement(dim, annotator_lst, idlst, df = merged_hEn, ):
    collection = {}
    for id_ in idlst:
        annotation_lst = []
        for annotator in annotator_lst:
            sub = df[(df.id == id_) & (df.annotator == annotator)]
            if len(sub)>0:
                annotation_lst.append(sub[dim].values[0])
            else:
                annotation_lst.append(np.nan)
            collection[id_] = annotation_lst
    return pd.DataFrame(collection, index = annotator_lst)     

def anno_res(df, exclude):
    merged_hEnidlst = df.id.unique()
    annotator_lst_hEn = df.annotator.unique()
    df_anno = pd.DataFrame()
    for dim in ['coherence', 'consistency', 'fluency', 'relevance']:
        cor_lst = agreement_return(dim, annotator_lst_hEn, merged_hEnidlst, df = df, exclude = exclude)
        df_anno_ = pd.DataFrame(cor_lst, columns = ["annotator1", "annotator2", "spearman", "kappa","annotation_count", "id_list"]) #"kappa",
        df_anno_["dimension"] = dim
        df_anno = pd.concat([df_anno, df_anno_])
    return df_anno

df = merged_hEn[(merged_hEn["sum"] >0) ] 
df_anno = anno_res(df, exclude = False)
df_anno[(df_anno.annotation_count > 10) & (df_anno.annotator2 != 6) ].groupby("dimension").mean()[["spearman"]].to_csv("./outputs/interannotation_agreement.csv")

human = pd.DataFrame(df.set_index("annotator").loc[[1,2,3]].groupby(["model_id", 'id']).mean()).reset_index()
chatty = df[df.annotator == 6]
# human chatty corr
human_chatty = pd.merge(human[['model_id', 'id', 'coherence', 'consistency', 'fluency', 'relevance',]], chatty[['model_id', 'id', 'coherence', 'consistency', 'fluency', 'relevance',]], on = ["id", "model_id"], how = "left", suffixes = ["_human", "_chatty"]).dropna()
human_chatty.corr("spearman").loc[["coherence_human", "consistency_human", "fluency_human", "relevance_human"], ["coherence_chatty", "consistency_chatty", "fluency_chatty", "relevance_chatty"]].to_csv("./outputs/hEn_human_chatty_agreement.csv")

merged_hEn = merged_hEn.rename(columns = {"rouge1": "ROUGE-1", "rougel":"ROUGE-L", 'bertscore_P': "BERTScore-P", 'bertscore_R':"BERTScore-R", 
                      'bertscore_F1': "BERTScore-F1", 'bartscore': "BARTScore", "moverscore":'MoverScore', "menli":'MENLI-W1',
                                    'MENLI_W0.8': 'MENLI-W.8', 'MENLI_W0.3':'MENLI-W.3', 'MENLI_W0.2':'MENLI-W.2', "DiscoScore_F":"DiscoScore-F"}) 
metric_diff = ['ROUGE-1', 'ROUGE-L', 'BERTScore-P', 'BERTScore-R', 'BERTScore-F1',
       'BARTScore', 'MoverScore', 'MENLI-W1','MENLI-W.8',  'MENLI-W.3', 'MENLI-W.2', 'DiscoScore-F', ]

df_ = merged_hEn[(merged_hEn.annotator != 6 )].set_index("id_model").loc[intersection_of_models]
inter1 = df_.groupby("id_model").mean()
caseALL = inter1.corr("spearman").iloc[:4, 5:19]
pval = inter1.corr(method=lambda x, y: spearmanr(x, y).pvalue).iloc[:4, 5:19] 
p = pval.applymap(lambda x: "\n\n" + ''.join(['*' for t in [.05, .01, .001] if x<=t]))
label = caseALL.round(2).astype(str) + p
plt.figure(figsize = (8,3))
X = caseALL.loc[:, metric_diff].round(2)
sns.heatmap(caseALL.loc[:,metric_diff].round(2),
            annot=label.loc[:,metric_diff],
            fmt="",
            cbar=False,
            linewidth=0.5,
            cmap="coolwarm_r")
plt.savefig('./outputs/hEn-mul-segment_cor.pdf',bbox_inches='tight', pad_inches=0 )  

