# HarshQA: Closed domain non-factoid Question Answering

An End-To-End Closed Domain Question Answering System designed for non-factoid questions.

HarshQA is a hybrid Q&A system composed of an enhanced Tf-Idf model and a Bert-ranker finetuned on passage similarity. The retriever uses query expansion and performs short text clustering. Bert is finally used to select and re-order the best candidate that might answer the input query.
The package also includes scikit learn wrappers for well known model such as Tf-Idf, Infersent (GloVe/fasText) and Bert pretrained.

![](https://github.com/wlambert01/harshQA/blob/master/images/background.png)
![Query Expansion](https://github.com/wlambert01/harshQA/blob/master/images/example0.png)
![Bert Reranker](https://github.com/wlambert01/harshQA/blob/master/images/example1.png)

## Table of Contents <!-- omit in toc -->

- [HarshQA Package](#Harsh-QA-Package)
- [Installation](#Installation)
  - [From source](#From-source)
  - [Download pretrained models and data](#Download-data)
  - [Hardware Requirements](#Hardware-Requirements)
  - [Installing Tika on Windows with a proxy](#Installing-Tika-on-Windows-with-a-proxy)
  - [Installing pattern](#Installing-pattern)
- [Getting started](#Getting-started)
  - [Preparing your data](#Preparing-your-data)
    - [Manual](#Manual)
  - [Training models](#Training-models)
  - [Making predictions](#Making-predictions)
  - [Call the whole QA pipeline from bash](#Call-the-whole-QA-pipeline-from-bash)
- [Evaluation on ESG Annual Reports](#Evaluation-on-ESG-Annual-Reports)
- [References](#References)

## HarshQA Package
-  👉 A pdf_to_text converter, that retrieves all sentences with a convenient structure.
-  👉 4 question answering models wrapped on a scikit learn predictor (including HarshQA model)
-  👉 model downloaders (GloVe, fasText, InferSent, Bert, Bert finetuned) and a script to finetune Bert 
-  👉 A Retriever module based on short text clustering, statistical semantics and query expansion
-  👉 A Bert Reranker  
-  👉 A Question Answering Pipeline that let you choose your model, your query, and more than 40 parameters ! 
-  👉 A notebook that contains our evaluations on pdf annual reports (3000 annotations)

## Installation

### From source

```shell
git clone https://github.com/harshQA-suite/harshQA.git
cd harshQA
pip install -e .
```

### Hardware Requirements

Experiments have been done with:

- **CPU** 👉 AWS EC2 `t2.medium` for predictions
- **GPU** 👉 AWS EC2 `t3.xlarge` for training the model

### Installing Tika on Windows with a proxy

Tika is simply the best solution to retrieve the content of a pdf files yet. Other libraries fail to handle linebreaks and manage different encodings. 

To install tika without proxies installed :
```bash
pip install tika
```
To install tika with a proxy you need to go through this steps:


### Installing pattern 

 To install the pattern NLP-features that we need you must grab the development branch:

```bash
sudo yum install mysql-server -y
sudo yum install mysql-devel 
git clone -b development https://github.com/clips/pattern
cd pattern
python setup.py install
```

### Downloading pre-trained models and data

You can download the pretrained models and data manually with our download functions:

```shell
from harshQA.utils.download import download_models, download_esg_data

directory = 'path-to-directory'

# Downloading data
download_esg_data(dir=directory)

# Downloading pre-trained BERT, finetuned Bert and Infersent models:
download_models(dir=directory)
```

You also need to download our finetuned Bert model that can be downloaded [here](https://drive.google.com/a/sage.biz/uc?export=download&confirm=D_hQ&id=1cyUrhs7JaCJTTu-DjFUqP6Bs4f8a6JTX)

## Getting started

### Preparing your data

#### Manual

To use `harshQA` you need to create a pandas dataframe with the following columns:

| directory_index             | paragraphs    | raw_paragraphs                                         |
| ----------------- | ------------------------------------------------------ | -------------------|
| Doc or Repo index | [Paragraph 1 of Article treated, ... , Paragraph N of Article treated] | [Paragraph 1 of Article raw, ... , Paragraph N of Article raw]|

#### With converters

The objective of `harshQA` converters is to make it easy to create this dataframe from your raw documents database. For instance the `pdfconverter` can create a `harshQA` dataframe from a list of directories containing `.pdf` files:

```python
from harshQA_pdf_reader.reader import pdfconverter

reader= pdfconverter(pdf_directories=['dir1','dir2'])
df=reader.transform()
```

You will need to install [Java OpenJDK](https://openjdk.java.net/install/) to use this converter together with [Tika](https://pypi.org/project/tika/) package (python) and the development branch of [Pattern](https://github.com/clips/pattern/tree/development). 



### Training models

Fit the pipeline on your corpus using the pre-trained reader:

```python
from harshQA_pdfreader.reader import pdfconverter
df = pdfconverter(['your_pdf_directories'])
harshQA_pipeline = QAPipeline(m5_args)
harshQA_pipeline.fit_reader()
harshQA_pipeline.fit()
```


### Making predictions

To get the best prediction given an input query:

```python
Questions=['Does Sanofi develop a risk management scheme to prevent industrial accidents and to ensure safety for all its workers']
Topics=[['risk management', 'safety workers', 'industrial accidents']]
harshQA_pipeline.predict(Question,Topics)
```

### Call the whole QA pipeline from bash

Given an input query csv file you can call the pipeline to make predictions and save results in an output_dir.

Query file structure:

| Query             | Topics    | 
| ----------------- | -------------------------------------| 
| Query 1 | [Topic 1 of Query 1, ... , Topic N of Query 1] | 
| Query 2 | [Topic 1 of Query 2, ... , Topic M of Query 2] |


Bash command lines:
```bash
      python harshQA.py \
      --demo=False\
      --model=5\
      --max_query_length=128\
      --domain="Chemicals"\
      --retrieved_company="Pfizer"\
      --query_dir="./utils/pdf_files/Tourism/Queries.csv"\
      --top_n=5\
      --transform_text=True\
      --size_cluster=50\
      --eval_batch_size=50\
      --pdf_directory="./utils/pdf_files/"\
      --whole_corpus="./utils/pdf_files/All"\
      --vocab_file="./data/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"\
      --vocab_builder="./corpusESG.json"\
      --w2v_path="./data/fastText/crawl-300d-2M.vec"\
      --model_path="./data/encoder/infersent2.pkl"\
      --init_checkpoint="./data/bert_msmarco/model.ckpt"\
      --bert_config_file="./data/bert_msmarco/bert_config.json"\
      --output_dir="./output"\
      --use_tpu=False
```
## Evaluation on ESG Annual Reports
!['ESG All Scores'](https://github.com/wlambert01/harshQA/blob/master/images/example3.png)
!['ESG Precision at 3'](https://github.com/wlambert01/harshQA/blob/master/images/example3.png)

## References

| Type                 | Title                                                                                                                                        | Author                                                                                 | Year |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---- |
| :newspaper: Paper    | [Short Text Clustering using Statistical Semantics ](http://www2015.wwwconference.org/documents/proceedings/companion/p805.pdf)                                                        | Ahmed K. Farahat, Fakhri Karray, Sepideh Seifzadeh, Mohamed S. Kamel                                   | 2015 |
| :newspaper: Paper    | [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)                                                                             | Rodrigo Nogueira, Kyunghyun Cho                                                        | 2019 |
| :video_camera: Video | [Stanford CS224N: NLP with Deep Learning Lecture 10 – Question Answering](https://youtube.com/watch?v=yIdF-17HwSk)                           | Christopher Manning                                                                    | 2019 |
| :newspaper: Paper    | [Attention is all you need](https://arxiv.org/abs/1706.03762)                                            | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,                                                                              | 2017 |
| :newspaper: Paper    | [A Multi-Resolution Word Embedding for Document Retrieval from Large Unstructured Knowledge Bases](https://arxiv.org/pdf/1902.00663.pdf)                                 | Tolgahan Cakaloglu, Xiaowei Xu                        | 2019 |
| :newspaper: Paper    | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)                         | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova                           | 2018 |
| :newspaper: Paper    | [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006)                                               | Noah A. Smith                                                                          | 2019 |
| :newspaper: Post    | [Breaking Down Bert](https://towardsdatascience.com/breaking-bert-down-430461f60efb)  | Shreya Ghelani | 2019 |
| :newspaper: Paper    | [Unsupervised Question Answering by Cloze Translation](https://arxiv.org/abs/1906.04980)                                                     | Patrick Lewis, Ludovic Denoyer, Sebastian Riedel                                       | 2019 |
| :computer: Framework | [PyTorch](https://arxiv.org/abs/1906.04980)                                                                                                  | Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan                               | 2016 |
| :computer: Framework | [PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers](https://github.com/huggingface/pytorch-pretrained-BERT) | Hugging Face                                                                           | 2018 |


