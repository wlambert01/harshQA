#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import wget

def download_models(dir="."):
    """
    Download Infersent and Bert models 
    Parameters
    ----------
    dir: str
        Directory where the dataset will be stored
    """

    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Download Infersent 2
    print("Downloading Infersent 2 model...")

    dir_squad11 = os.path.join(dir, "Infersent2")
    squad11_urls = ["https://dl.fbaipublicfiles.com/infersent/infersent2.pkl",
                    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"]

    if not os.path.exists(dir_squad11):
        os.makedirs(dir_squad11)

    for squad_url in squad11_urls:
        file = squad_url.split("/")[-1]
        if os.path.exists(os.path.join(dir_squad11, file)):
            print(file, "already downloaded")
        else:
            wget.download(url=squad_url, out=dir_squad11)

    # Download Bert Base Uncased pretrained

    print("\nDownloading Bert Uncased pretrained model data...")

    dir_squad20 = os.path.join(dir, "Bert_pretrained")
    squad20_urls = ["https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"]

    if not os.path.exists(dir_squad20):
        os.makedirs(dir_squad20)

    for squad_url in squad20_urls:
        file = squad_url.split("/")[-1]
        if os.path.exists(os.path.join(dir_squad20, file)):
            print(file, "already downloaded")
        else:
            wget.download(url=squad_url, out=dir_squad20)

def download_esg_data(dir="."):
    """
    Download esg annual report dataset
    Parameters
    ----------
    dir: str
        Directory where the dataset will be stored
    """

    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    url = "https://github.com/harshQA/blob/data_esg-v1.1.csv"

    print("\nDownloading esg data...")

    file = url.split("/")[-1]
    if os.path.exists(os.path.join(dir, file)):
        print(file, "already downloaded")
    else:
        wget.download(url=url, out=dir)

