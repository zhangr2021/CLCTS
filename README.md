# CLCTS
(work in progress)
This repository contains the code and data for our paper: [Cross-lingual Cross-temporal Summarization:
Dataset, Models, Evaluation](link_to_arxiv).

> **Abstract**: 
> While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility, information sharing, and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate task finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find our intermediate task finetuned e2e models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and correlates moderately with human evaluations though it is prone to giving lower scores. ChatGPT also seems to be very adept at normalizing historical text. We finally test ChatGPT in a scenario with
adversarially attacked and unseen source documents and find that ChatGPT is better at omission and entity swap than negating against its prior knowledge and it generates summaries of mediocre quality from unseen source documents..



## 🚀 CLCTS Corpus

We release our [CLCTS datasets](path/to/data). 

## 🚀 Experiments
To reproduce the evaluations conducted in this work, please check the folder [experiments](experiments).


If you use the code or data from this work, please include the following citation:


If you have any questions, feel free to contact us!

Ran Zhang ([ran.zhang@uni-bielefeld.de](mailto:ran.zhang@uni-bielefeld.de)), Jihed Ouni ([jihed.ouni@stud.tu-darmstadt.de](mailto:jihed.ouni@stud.tu-darmstadt.de)) and Steffen Eger ([steffen.eger@uni-bielefeld.de](mailto:steffen.eger@uni-bielefeld.de))

Check our group page ([NLLG](https://nl2g.github.io/)) for other ongoing projects!
