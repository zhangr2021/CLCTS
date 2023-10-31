# CLCTS
(work in progress)
This repository contains the code and data for our paper: [Cross-lingual Cross-temporal Summarization:
Dataset, Models, Evaluation]([link_to_arxiv](https://arxiv.org/abs/2306.12916)).

> **Abstract**: 
> While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find that our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations but is prone to giving lower scores. ChatGPT also seems very adept at normalizing historical text and outperforms context-unaware spelling normalization tools such as Norma. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT profits from its prior knowledge to a certain degree, with better performances for omission and entity swap than negation against its prior knowledge. This benefit inflates its assessed quality as ChatGPT performs slightly worse for unseen source documents compared to seen documents. We additionally introspect our models' performances to find that longer, older and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task.

## CLCTS Corpus

We release our [CLCTS datasets](dataset/CLCTS_corpus). 

## Experiments
To reproduce the evaluations conducted in this work, please check the folder [results for section datasets](results/section_datasets) and [results for section experiments](results/section_experiments).

We provide both human and ChatGPT evaluation results together with automatic metric evaluation for [hDe](results/section_experiments/hDe) and [hEn](results/section_experiments/hEn). 

# Citation
If you use the code or data from this work, please include the following citation:
```
@article{zhang2023cross,
  title={Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation},
  author={Zhang, Ran and Ouni, Jihed and Eger, Steffen},
  journal={arXiv preprint arXiv:2306.12916},
  year={2023}
}
```

# Contacts
If you have any questions, feel free to contact us!

Ran Zhang ([ran.zhang@uni-bielefeld.de](mailto:ran.zhang@uni-bielefeld.de)), Jihed Ouni ([jihed.ouni@stud.tu-darmstadt.de](mailto:jihed.ouni@stud.tu-darmstadt.de)) and Steffen Eger ([steffen.eger@uni-bielefeld.de](mailto:steffen.eger@uni-bielefeld.de))

