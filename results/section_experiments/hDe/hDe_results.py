import pandas as pd
import os
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score 
import scipy

output_all_hDe = pd.read_csv("./eval_metric_hDe.csv")
merged_hDe = pd.read_csv("hDe-human_chatGPT_annotation.csv")
merged_hDe.annotator = merged_hDe.annotator.astype(str)
print("output shape:", output_all_hDe.shape[0], "annotation output: ", merged_hDe.shape[0])

order_metric = ['rouge1', 'rougel', 'bertscore_P', 'bertscore_R', 'bertscore_F1',  'bartscore','moverscore','menli', 
                'MENLI_W0.8', 'MENLI_W0.3', 'MENLI_W0.2', "DiscoScore_F", "DiscoScore_S"]

model_order = ['German_25_False', 'German_25_True', 'German_100', '1','3', '2', '4','9_2', '9_1',  '10',  'chatGPT_title', 
               'chatGPT_e2e', 'chatGPT_pipeline',]

filtered = output_all_hDe.set_index("id").loc[output_all_hDe[output_all_hDe.model_id == "1"].id]
latex = filtered.groupby("model_id").mean().iloc[:, -14:].loc[model_order, order_metric]
def turn_string(row):
    lst = ["{:.3f}".format(i.round(3)) for i in row.values]
    lst_ = ["/".join(lst[:2]), "/".join(lst[2:5]), lst[5], lst[6], "/".join(lst[7:-2]), lst[-2]]
    print(lst_)
    return "&".join(lst_)

pd.DataFrame(latex.apply(lambda x: turn_string(x), axis = 1)).to_csv("outputs/latex_experiment_hDe-monolingual.csv", index = True)

merged_hDe["id_model"] = merged_hDe.apply(lambda x: str(x["id"]) + "_" + x["model_id"], axis = 1)
intersection_of_models = set(merged_hDe[(merged_hDe.annotator != "8")].id_model.unique()).intersection(set(merged_hDe[(merged_hDe.annotator == "8" )].id_model.unique()))

# human annotation result avg. 
print("human annotaiton All")
summary_level = pd.DataFrame(merged_hDe[(merged_hDe.annotator != "8")].set_index("id_model").loc[intersection_of_models].groupby(["model_id", "id"]).mean()).reset_index() 
human_mean = summary_level.groupby("model_id").mean().iloc[:,1:5].loc[model_order]

# average ChatGPT annotation results
summary_level = pd.DataFrame(merged_hDe[merged_hDe.annotator == "8"].set_index("id_model").loc[intersection_of_models].groupby(["model_id", "id"]).mean()).reset_index() 
chatGPT_mean = summary_level.groupby("model_id").mean().iloc[:,1:5].loc[model_order]

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
pd.DataFrame(annotation_human_chatty.apply(lambda x: turn_string(x) + "\\" + "\\", axis = 1)).to_csv("outputs/hDe_latex_human_chatty_anno.csv", index = True)

def agreement_return(dim, annotator_lst_hEn, merged_hEnidlst, df):
    tmp = agreement(dim, annotator_lst_hEn, merged_hEnidlst, df)
    cor_lst = []
    for a1 in tmp.index:
        for a2 in  tmp.index:
            if int(a2)>int(a1):
                agree = tmp.loc[[str(a1),str(a2)]].dropna(axis='columns')
                if len(agree.loc[str(a1)]) > 0:
                    cor = spearmanr(agree.loc[str(a1)], agree.loc[str(a2)]).correlation
                    #print(cor, agree, "\n\n\n\n")
                    k1 = [int(val) for val in agree.loc[str(a1)]]
                    k2 = [int(val) for val in agree.loc[str(a2)]]
                    kappa = cohen_kappa_score(k1, k2)
                    if a2 == 8:
                        print(agree)
                    cor_lst.append((a1, a2, cor, kappa, len(agree.loc[str(a1)]), agree.columns.tolist())) 
                else: print("length problem", agree)
    return cor_lst

def agreement(dim, annotator_lst,idlst, df = merged_hDe, ):
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

def anno_res(df):
    merged_hEnidlst = df.id.unique()
    annotator_lst_hEn = df.annotator.unique()
    df_anno = pd.DataFrame()
    for dim in ['coherence', 'consistency', 'fluency', 'relevance']:
        cor_lst = agreement_return(dim, annotator_lst_hEn,merged_hEnidlst, df = df)
        df_anno_ = pd.DataFrame(cor_lst, columns = ["annotator1", "annotator2", "spearmanr", "kappa","annotation_count", "id_list"]) #"kappa",
        df_anno_["dimension"] = dim
        df_anno = pd.concat([df_anno, df_anno_])
    return df_anno

df = merged_hDe[(merged_hDe["sum"] >0) ] 
df_anno = anno_res(df,)

df_anno[(df_anno.annotation_count > 10) & (df_anno.annotator2 != "8")].groupby("dimension").mean().iloc[:,:1].to_csv("outputs/interannotation_agreement.csv")

human = pd.DataFrame(df[(df.annotator != "8")].groupby(["model_id", 'id']).mean()).reset_index()
chatty = df[df.annotator == "8"]
# human chatty corr
human_chatty = pd.merge(human[['model_id', 'id', 'coherence', 'consistency', 'fluency', 'relevance',]], chatty[['model_id', 'id', 'coherence', 'consistency', 'fluency', 'relevance',]], on = ["id", "model_id"], how = "left", suffixes = ["_human", "_chatty"]).dropna()

human_chatty.corr("spearman").loc[["coherence_human", "consistency_human", "fluency_human", "relevance_human"], ["coherence_chatty", "consistency_chatty", "fluency_chatty", "relevance_chatty"]].to_csv("outputs/hDe_human_chatty_agreement.csv")

merged_hDe = merged_hDe.rename(columns = {"rouge1": "ROUGE-1", "rougel":"ROUGE-L", 'bertscore_P': "BERTScore-P", 
                                          'bertscore_R':"BERTScore-R", 'bertscore_F1': "BERTScore-F1", 'bartscore': "BARTScore", 
                                          "moverscore":'MoverScore', "menli":'MENLI-W1', 'MENLI_W0.8': 'MENLI-W.8', 
                                          'MENLI_W0.3':'MENLI-W.3', 'MENLI_W0.2':'MENLI-W.2', "DiscoScore_F":"DiscoScore-F"}) 

merged_hDe["id_model"] = merged_hDe.apply(lambda x: str(x["id"]) + "_" + x["model_id"], axis = 1)
intersection_of_models = set(merged_hDe[(merged_hDe.annotator != "8" ) ].id_model.unique()).intersection(set(merged_hDe[(merged_hDe.annotator == "8" )].id_model.unique()))
metric_diff = ['ROUGE-1', 'ROUGE-L', 'BERTScore-P', 'BERTScore-R', 'BERTScore-F1',
       'BARTScore', 'MoverScore', 'MENLI-W1','MENLI-W.8',  'MENLI-W.3', 'MENLI-W.2', 'DiscoScore-F', 
      ]

pd.options.display.float_format = '{:,.2f}'.format
df_ = merged_hDe[(merged_hDe.annotator != "8" )].set_index("id_model").loc[intersection_of_models]
inter1 = df_.groupby("id_model").mean()
caseALL = inter1.corr("spearman").iloc[:4, 5:19]
pval = inter1.corr(method=lambda x, y: spearmanr(x, y).pvalue).iloc[:4, 5:19] 
p = pval.applymap(lambda x: "\n\n" + ''.join(['*' for t in [.05, .01, .001] if x<=t]))
label = caseALL.round(2).astype(str) + p

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (8,3))
sns.heatmap(caseALL.loc[:,metric_diff].round(2),
            annot=label.loc[:,metric_diff],
            fmt="",
            cbar=False,
            linewidth=0.5,
            cmap="coolwarm_r")
plt.savefig('outputs/hDe-en-segment_cor.pdf',bbox_inches='tight', pad_inches=0 )  





