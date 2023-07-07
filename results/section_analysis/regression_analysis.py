# regress against standardized 
import pandas as pd
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_vif(df, features):    
    # source https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


def merge_df(mdd_df, stanza, embedding, hDf, dataset, bins):
    mdd_df = pd.merge(mdd_df, stanza, on = "sentence", how = "left")
    mdd_df = mdd_df[(mdd_df.dataset == dataset) & (mdd_df.col == "text") & (mdd_df.n_tokens > 1)].groupby("id").mean()[["dependency_distance_mean"]].reset_index()
    
    embedding = embedding[embedding.dataset == dataset].groupby("id").mean()[["similarity"]].reset_index()
    embedding.id = embedding.id.astype(int)
    
    hDf["year"] = pd.cut(hDf['Year'], bins=bins, labels = list(range(len(bins)-1)))
    hDf = pd.merge(mdd_df, hDf, on = "id", how = "right")
    hDf = pd.merge(hDf, embedding, on ="id", how = "left")
    
    return mdd_df, embedding, hDf

from statsmodels.formula.api import ols

def prepare_reg(hDf, output_all, model_order):
    reg_matrix = pd.merge(hDf, output_all, on = "id")
    reg = reg_matrix[['dependency_distance_mean', 'similarity', 'year', 'token_doc', 'model_id', 'bertscore_F1']]
    
    tmp = np.log(reg[['dependency_distance_mean', 'similarity', 'token_doc', 'bertscore_F1']])
    tmp.columns = ['dependency_distance_mean_log', 'similarity_log', 'token_doc_log', 'bertscore_F1_log']
    reg = pd.concat([reg, tmp], axis = 1)
    reg = reg.set_index("model_id").loc[model_order].reset_index()
    
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    matrix = ss.fit_transform(reg[['dependency_distance_mean', 'similarity', 'token_doc', 'bertscore_F1']])
    st = pd.DataFrame(matrix, columns =['dependency_distance_mean_st', 'similarity_st', 'token_doc_st', 'bertscore_F1_st'])
    reg = pd.concat([reg, st], axis = 1)
    
    ss = StandardScaler()
    st_log_matrix = ss.fit_transform(reg[['dependency_distance_mean_log', 'similarity_log', 'token_doc_log', 'bertscore_F1_log']])
    st_log = pd.DataFrame(matrix, columns =['dependency_distance_mean_st_l', 'similarity_st_l', 'token_doc_st_l', 'bertscore_F1_st_l'])
    reg = pd.concat([reg, st_log], axis = 1)
    
    return reg

def ols_(reg):
    #normalization
    fit = ols('bertscore_F1_st ~ C(year) + C(model_id) + similarity_st + token_doc_st + dependency_distance_mean_st', data=reg).fit() 
    print(fit.summary())
    df_st = pd.concat([fit.params, fit.pvalues], axis = 1)
    
    fit = ols('bertscore_F1_log ~ C(year) + C(model_id) + similarity_log + token_doc_log + dependency_distance_mean_log', data=reg).fit() 
    #print(fit.summary())
    df_log = pd.concat([ fit.params, fit.pvalues], axis = 1)
    return df_st, df_log

embedding = pd.read_csv("../section_datasets/embedding_distance/all_datasets_embedding_distance.csv")
hDf = pd.read_csv("../../dataset/CLCTS_corpus/CLCTS_hDe.csv")
output_all = pd.read_csv("../section_experiments/hDe/eval_metric_hDe.csv")
mdd_df = pd.read_pickle("../section_datasets/statistics_german_sent.pkl")
stanza = pd.read_csv("../section_datasets/parsing/german_parsing_result_stanford_all.csv")

model_order_hDe = ['German_25_False', 'German_25_True', 'German_100', '1','3', '2', '4','9_2', '9_1',  '10',  'chatGPT_title', 
               'chatGPT_e2e', 'chatGPT_pipeline',]

model_order_hEn = ['English_25_False', 'English_25_True', 'English_100', "1", "2", "3", "4", "5", "6", "11", 
               'chatGPT_title', "chatGPT_e2e",  "chatGPT_pipeline"]

variable = ["year", "token_doc", "dependency_distance_mean", "similarity"]

_,_,hDf = merge_df(mdd_df = mdd_df, stanza = stanza, embedding = embedding, hDf = hDf, dataset = "CLCT_hDe", bins =[1800, 1850, 1900])
calculate_vif(hDf[variable].dropna(), variable).round(2)

reg = prepare_reg(hDf = hDf, output_all = output_all, model_order = model_order_hDe)
df_st_hDe, _ = ols_(reg)

df_st_hDe = df_st_hDe.rename(columns = {0: "coefficient", 1: "pvalues"})
pd.options.display.float_format = '{:,.2f}'.format

p = df_st_hDe[["pvalues"]].applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
label = df_st_hDe[["coefficient"]].round(2).astype(str), p

coef_hDe = pd.concat([df_st_hDe[["coefficient"]].round(2).astype(str), p], axis = 1)
coef_hDe["content"] = coef_hDe.apply(lambda x: " ".join([x["coefficient"], x["pvalues"]]), axis = 1)
coef_hDe.sort_values("coefficient").to_csv("outputs/hDe-regression.csv")


hDf = pd.read_csv("../../dataset/CLCTS_corpus/CLCTS_hEn.csv")
output_all = pd.read_csv("../section_experiments/hEn/eval_metric_hEn.csv")
mdd_df = pd.read_pickle("../section_datasets/statistics_english_sent.pkl")
stanza = pd.read_csv("../section_datasets/parsing/english_parsing_result_stanford_all.csv")

bins = [1600, 1800, 1850, 1900]
_,_,hDf = merge_df(mdd_df = mdd_df, stanza = stanza, embedding = embedding, hDf = hDf, dataset = "CLCT_hEn", bins = bins)
calculate_vif(hDf[variable].dropna(), variable).round(2)

reg = prepare_reg(hDf = hDf, output_all = output_all, model_order = model_order_hEn)
df_st_hEn, _ = ols_(reg)

pd.options.display.float_format = '{:,.2f}'.format
df_st_hEn = df_st_hEn.rename(columns = {0: "coefficient", 1: "pvalues"})
p = df_st_hEn[["pvalues"]].applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
label = df_st_hEn[["coefficient"]].round(2).astype(str), p
coef = pd.concat([df_st_hEn[["coefficient"]].round(2).astype(str), p], axis = 1)
coef["content"] = coef.apply(lambda x: " ".join([x["coefficient"], x["pvalues"]]), axis = 1)
coef.sort_values("coefficient").to_csv("outputs/hEn-regression.csv")

de_index = ['Intercept', 'C(year)[T.1]', 'C(model_id)[T.German_25_False]', 'C(model_id)[T.German_25_True]', 
 'C(model_id)[T.German_100]', 'C(model_id)[T.2]', 'C(model_id)[T.3]', 'C(model_id)[T.4]', 'C(model_id)[T.9_2]', 
 'C(model_id)[T.9_1]',  'C(model_id)[T.10]', 'C(model_id)[T.chatGPT_title]',  'C(model_id)[T.chatGPT_e2e]', 
 'C(model_id)[T.chatGPT_pipeline]', 'similarity_st', 'token_doc_st', 'dependency_distance_mean_st']
coef_hDe.loc[de_index].to_csv("outputs/df_st_hDe.csv")

en_index = ['Intercept', 'C(year)[T.1]', 'C(year)[T.2]', 'C(model_id)[T.English_25_False]', 'C(model_id)[T.English_25_True]',
'C(model_id)[T.English_100]', 'C(model_id)[T.2]', 'C(model_id)[T.3]', 'C(model_id)[T.4]', 'C(model_id)[T.5]', 
 'C(model_id)[T.6]',  'C(model_id)[T.11]', 'C(model_id)[T.chatGPT_title]',  'C(model_id)[T.chatGPT_e2e]', 
 'C(model_id)[T.chatGPT_pipeline]', 'similarity_st', 'token_doc_st', 'dependency_distance_mean_st']
coef.loc[en_index].to_csv("outputs/df_st_hEn.csv")

