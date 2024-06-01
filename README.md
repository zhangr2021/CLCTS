# CLCTS:  Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation
This repository contains the code and data for our paper: [Cross-lingual Cross-temporal Summarization:
Dataset, Models, Evaluation](https://arxiv.org/abs/2306.12916).

> **Abstract**: 
> While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We (1) build the first CLCTS corpus with 328 instances for hDe-En (extended version with 455 instances) and 289 for hEn-De (extended version with 501 instances), leveraging historical fiction texts and Wikipedia summaries in English and German; (2) examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks; (3) explore the potential of GPT-3.5 as a summarizer; (4) report evaluations from humans, GPT-4, and several recent automatic evaluation metrics. Our results indicate that intermediate task finetuned end-to-end models generate bad to moderate quality summaries while GPT-3.5, as a zero-shot summarizer, provides moderate to good quality outputs. GPT-3.5 also seems very adept at normalizing historical text. To assess data contamination in GPT-3.5, we design an adversarial attack scheme in which we find that GPT-3.5 performs slightly worse for unseen source documents compared to seen documents. Moreover, it sometimes hallucinates when the source sentences are inverted against its prior knowledge with a summarization accuracy of 0.67 for plot omission, 0.71 for entity swap, and 0.53 for plot negation. Overall, our regression results of model performances suggest that longer, older, and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task. Regarding evaluation, we observe that both GPT-4 and BERTScore correlate moderately with human evaluations, implicating great potential for future improvement.

## CLCTS Corpus

We release our [CLCTS datasets](dataset/CLCTS_corpus). 

## Experiments
To reproduce the evaluations conducted in this work, please check the folder [results for section datasets](results/section_datasets) and [results for section experiments](results/section_experiments).

We provide both human and ChatGPT evaluation results together with automatic metric evaluation for [hDe](results/section_experiments/hDe) and [hEn](results/section_experiments/hEn). 

## Citation
If you use the code or data from this work, please include the following citation:
```
@article{zhang2024cross,
  title={Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation},
  author={Zhang, Ran and Ouni, Jihed and Eger, Steffen},
  journal={Computational Linguistics},
  pages={1--44},
  year={2024},
  publisher={MIT Press 255 Main Street, 9th Floor, Cambridge, Massachusetts 02142, USA}
}
```

## Contacts
If you have any questions, feel free to contact us!

Ran Zhang ([ran.zhang@uni-mannheim.de](mailto:ran.zhang@uni-mannheim.de)), Jihed Ouni ([jihed.ouni@stud.tu-darmstadt.de](mailto:jihed.ouni@stud.tu-darmstadt.de)) and Steffen Eger ([steffen.eger@uni-mannheim.de](mailto:steffen.eger@uni-mannheim.de))

