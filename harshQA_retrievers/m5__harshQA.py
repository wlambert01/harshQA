#!/usr/bin/env python
# coding: utf-8

import warnings
from utils.utils  import hide_warn
warnings.warn=hide_warn
import torch
import torch.nn as nn
import tensorflow as tf 
import pandas as pd
import numpy as np
import json 

#Local dependencies
from harshQA_reranker.tokenization import* 
import harshQA_reranker.metrics as metrics
import harshQA_reranker.modeling as modeling
import harshQA_reranker.optimization as optimization 
from harshQA_retrievers.m3__Tfidf import m3_Tfidf
from harshQA_reranker.harshQA_tfrecord import *
from harshQA_reranker.harshQA_bert_builder import * 
from harshQA_reranker.harshQA_run_msmarco import run_msmarco,msmarco_args
from sklearn.base import BaseEstimator
from tabulate import tabulate

#Utils
from utils.utils import remove_non_alpha
from utils.utils import generate_querries


class m5_harshQA(BaseEstimator):
    """
    A scikit-learn estimator composed of short text clustering, query expansion and a Bert retriever finetuned on reranking tasks.

    Trains a tf-idf matrix from a corpus, apply Farahat and Nystrom method suggested in their research paper
    http://www2015.wwwconference.org/documents/proceedings/companion/p805.pdf
    
    It first select a set of important words within the whole corpus and then it builds a sentence similarity kernel.
    The kernel is computed by computing a low rank approximation  of the correlation matrix between terms, using Nystrom method. (see below)
    http://proceedings.mlr.press/v5/kumar09a/kumar09a.pdf
    
    For querry treatment we use two features:

    - Word importance function that is provided in the InferSent class, based on maxpooling of stacked forward and backwards lstm output.
    We modify Infersnent function to normalize importance scores to be between O and 1, and we retrieved the top importance words until a given threshold of cumulative importance.

    -Query expansion, using the term-term correlation matrix fitted on the whole corpus. 
    It takes one or multiple topic(s) within your query and performs query expansion by finding the most pertinent synonyms for this topic(s).
    This is key, especially when your query is large, since it will increase the importance of a given topic for the retriever module to perform better.
    It also greatly decrease the risk of missing potential answers since it disminish synonym sensibility 

    The retriever of harshQA is a Tf-Idf retriever that used the document similariy kernel and query expansion.
    The retriever returns a cluster of potential answers that will be used to feed bert reranker. 

    Then it applies the Bert reranker on this cluster of selected documents and return the most top_N similar document by taking the dot product
     of the vectorized input document and enhaned tf-idf embedding matrix.

    Our code provides also a snippet to fine-tune Bert on a reranking task and sentence similarity with transfer learning (see harshQA_run_msmarco) .
    You can also download a pretrained model here: https://drive.google.com/file/d/1crlASTMlsihALlkabAQP6JTYIZwC1Wm8/view
   
    Parameters
    ------------------------------
    rk : int, optional
        [rank of the approximation for term to term correlation matrix] (the default is 500)
    top_n : int
        maximum number of top articles to retrieve
        header should be of format: title, paragraphs.
    verbose : bool, optional
        If true, all of the warnings related to data processing will be printed.
    whole_corpus: str 
        The whole corpus to fit before predicting, smaller corpus, i.e the corpus
        to build and save the tfidfs weights, vocab and semantic kernel.
    vocab_builder : str 
        The path to build our own vocabulary file for the pdf-reader module.
        The vocabulary file that the BERT model was trained on.
    w2v_path : str 
        Path of the .vec file of GloVe or FastText.
    model_path : str
        Path of the .pkl file of infersent model.
    output_dir : str 
        The output directory where all the model bert,tfidf checkpoints will be written after train 
        Will also store the raw tsv and tfrecords predictions.
    bert_config_file : str
            This specifies the model architecture.
    init_checkpoint : str
        Initial checkpoint (usually from a pre-trained BERT model).

    size_cluster :  int
        Size of the clusters of candidate to feed in neural network.
    max_seq_length : int 
        The maximum total input sequence length after WordPiece tokenization. 
        Sequences longer than this will be truncated, and sequences shorter than this will be padded.


    min_gram : int, Min grams to use for tf-idf model retriever.
    max_gram : int,  Max grams to use for tf-idf model retriever.

    max_query_length : int,  The maximum query sequence length after WordPiece tokenization. 
                    "Sequences longer than this will be truncated.
    msmarco_output : bool,  Whether to write the predictions to a MS-MARCO-formatted file.
    do_train : bool,  Whether to run training.
    do_eval : bool, Whether to predict or not.
    train_batch_size : int, Total batch size for training, should be less lr equal than size_cluster.
    eval_batch_size : int, Total batch size for eval should be less  or equal than size_cluster and divide size_cluster
    learning_rate: float, The initial learning rate for Adam.
    num_train_steps : int, Total number of training steps to perform.
    max_eval_examples : int, Maximum number of examples to be evaluated.
    num_warmup_steps :  int, Number of training steps to perform linear learning rate warmup.
    save_checkpoints_steps : int, How often to save the model checkpoint.
    iterations_per_loop: int, How many steps to make in each estimator call.

    use_tpu : bool, Whether to use TPU or GPU/CPU.
    tpu_name : str, The Cloud TPU to use for training. This should be either the name used when creating 
                    the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url."
    tpu_zone : str,  GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect
                     the GCE project from  metadata.
    gcp_project : str,  Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically 
                        detect the GCE project from metadata "
    master : str,  TensorFlow master URL.
    num_tpu_cores : int, Only used if `use_tpu` is True. Total number of TPU cores to use.



    Examples
    -------------------------

    >>> retriever = m5_harshQA(top_n=5, 
                            size_cluster=50, 
                            eval_batch_size=50,
                            w2v_path='./data/fastText/crawl-300d-2M.vec', 
                            model_path='./data/encoder/infersent2.pkl',
                            output_dir='./output/',
                            vocab_file='./data/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt',
                            bert_config_file='./data/bert_msmarco/bert_config.json',
                            init_checkpoint='./data/bert_msmarco/model.ckpt')
    
    >>> retriever.fit(X=df['content'])

    >>> doc_index=int(input('Which document do you want to query?'))
    >>> retriever.transform(X=df.loc[doc_index,'content'])
    
    >>> Query=[' Has the company formalized  a strategy to reduce its greenhouse gas emissions?' , 
                'Does the company have an emergency plan to cope with the consequences of industrial accidents and to ensure business continuity?']

    >>> Topics=[['strategy reduce emissions'], ['emergency plan', 'industrial accidents', 'business continuity']]

    >>> closest_docs,scores = self.retriever.predict(Query,Topics)

    """


    def __init__(self,
        w2v_path,
        model_path,
        output_dir,
        vocab_file,
        bert_config_file,
        init_checkpoint,
        max_query_length=128,
        max_seq_length=512,
        top_n=5,
        size_cluster=100,
        eval_batch_size=100,
        train_batch_size=100,
        do_eval=True,
        do_train=False,
        max_eval_examples=None,
        msmarco_output=True,
        save_kernel_idx_path=None,
        save_kernel_path=None,
        save_kernel_vocab_path=None,
        save_idfs_path=None,
        save_features_path=None,
        verbose=False,
        save_checkpoints_steps=100,
        num_warmup_steps=4000,
        num_train_steps=400000,
        iterations_per_loop=10,
        threshold_w=0.002,
        portion_w=1.0,
        rank=500,
        learning_rate=1e-6,
        num_tpu_cores=8,
        master=None,
        gcp_project=None,
        use_tpu=False,
        tpu_name=None,
        tpu_zone=None):

        kwargs=locals()

        #Parameters to tune
        self.size_cluster=size_cluster
        self.max_query_length=max_query_length
        self.max_seq_length=max_seq_length
        self.top_n = top_n
        self.threshold_importance=threshold_w
        self.rk=rank
        self.prop_w=portion_w
        self.verbose =verbose
        self.vocab_file=vocab_file
        self.Infersent_wpath=w2v_path
        self.Infersent_mpath=model_path
        self.output_dir=output_dir
        self.kernel_idx_path=save_kernel_idx_path
        self.kernel_path=save_kernel_path
        self.kernel_vocab_path=save_kernel_vocab_path
        self.save_idfs_path=save_idfs_path
        self.save_features_path=save_features_path

        
        #Parameters for pretrain tf-idf and querry generator
        
        self.kwargs_Tf_Idf = {key: value for key, value in kwargs.items()
                           if (key in m3_Tfidf.__init__.__code__.co_varnames and key!='self')}
        self.kwargs_run_msmarco={key: value for key, value in kwargs.items()
                           if (key in msmarco_args.__init__.__code__.co_varnames and key!='self')}
        self.dic_emb={}


    def select_term_Nystrom(self,T,prop,threshold,stemmed_content,terms):

        #T is the Tf-Idf Matrix 
        #s is the fraction of important words that we want to retrieve (s=1.0 generally)
        #Retrieve all the stems of tf-idf vocabulary
        
        #First we remove digits of the terms candidate
        n=int(prop*len(terms))
        idx_words=[]
        for i,term in enumerate(terms):
            try: float(term)
            except: idx_words.append(i)

        #Secondly we set a threshold to select words that appear at least each p documents

        #Set threshold to select words that appear at least each p documents 
        self.freq_term=np.zeros_like(T)
        
        for j,stem in enumerate(stemmed_content):
            for i,c in enumerate(terms):
                if c in stem:
                    self.freq_term[j,i]+=1
                    
        self.freq_term_by_doc=np.apply_along_axis(lambda x: np.mean(x),0,self.freq_term)
        ids=np.where(self.freq_term_by_doc>threshold)[0]
        idx_words=[ i for i in ids if i in idx_words]


        #Init selection of terms with most correlation 
        if prop!=1.0:
            prob=np.sum(np.abs(T)>0,axis=0)
            prob=prob/np.sum(prob)
            prob=np.squeeze(prob)
            idx=[]
            m=len(idx_words)

            for i in range(n):

                p=int(np.random.choice(m,1,prob[0]))
                idx.append(int(idx_words[p]))
                prob[idx_words[p]]=0
                prob=prob/np.sum(prob)

            return idx
        
        else:
            idx=idx_words
            return idx
    
    

    def fit(self, X=None, y=None): 

        self.retriever = m3_Tfidf(**self.kwargs_Tf_Idf)
        self.retriever.fit(X)
        self.retriever.transform(X)
        
        if self.kernel_path==None or self.kernel_idx_path==None:
            
            TF_emb=self.retriever.tfidf_matrix
            X_stemmed=[self.retriever.tokenize(s) for s in X]
            self.vocab=self.retriever.vectorizer.get_feature_names()

            self.idx=self.select_term_Nystrom(TF_emb.toarray(),self.prop_w,self.threshold_importance,X_stemmed,self.vocab)   
            if self.rk==-1:
                self.rk=len(self.idx)

            # Generate low rank approximation of the Kernel with Nystrom Farahat approach 
            TF=TF_emb.toarray().transpose()
            L=np.eye(TF.shape[0])*np.sqrt(TF.shape[0])
            L_inv=np.linalg.inv(L)
            G=L_inv@TF@TF.transpose()@L_inv
            Gs=G[self.idx,:]
            Gs=Gs[:,self.idx]
            S,V,D=np.linalg.svd(Gs, full_matrices=True)
            Ssub,Vsub=S[:,:self.rk], np.diag(V)[:self.rk,:self.rk]
            D_sub_inv=np.diag(np.apply_along_axis(lambda x: 1/np.sqrt(x) , 0, V[:self.rk]))
            self.kernel=(((D_sub_inv@Ssub.transpose())@TF[self.idx,:])@(TF.transpose()))
            if self.kernel_path!=None:
                np.save(self.kernel_path,self.kernel)
                voc=np.array(self.vocab)[self.idx]
                with open(self.kernel_vocab_path, 'w') as outfile:
                    json.dump(list(voc),outfile)
            print("semantic kernel has been computed and saved")
            
        else:
            self.idx=json.load(open(self.kernel_idx_path, mode = 'rb'))
            self.kernel=np.load(self.kernel_path)
            print("semantic kernel has been retrieved ")
        return self


    def transform(self,X=None): #generate new enhanced tf-idf-farahat features ( co-occurence weight frequency matrix) 

        """ 
        X should be a list containing the list of treated text and the list of  untreated text, otherwise it will duplicate the sentences provided
        """
        
        self.content_repo=X #format [[text_treated],[text_raw]]
        self.content=X[0] #grab the treated text 
        self.retriever.transform(self.content)
        TF_docs=self.retriever.tfidf_matrix.toarray().transpose()
        emb_doc=self.kernel@TF_docs
        self.docs_emb=np.apply_along_axis(lambda x: x/np.sqrt(np.sum(x**2)), 1,emb_doc.transpose())
        return self
    
    def find_synonyms(self,word,all_words,ktop=3,stopwords=[]):
        
        kern=self.kernel
        tokenizer=self.retriever.tokenize
        
        #"Preprocess text"
        word=word.strip()
        lw=word.split()
        lw=[tokenizer(w)[0] for w in lw]
        stopwords=[tokenizer(w)[0] for w in stopwords]
        assert len(lw)<=3,'there is too much words in you important terms'

        #"Initiate ids ; voc and structures to grab most pertinent synonyms "
        ids_=np.array(self.idx)
        voc=np.array(all_words)[ids_]
        dic_ids={}
        dic_syns={}
        output=[]

        #If a one word topic is given then catch the ktop best synonyms in the corpus (the stems)'
        if len(lw)==1:
        
            w=lw[0]
            id_kernel=[i for i,c in enumerate(voc) if c==w]

            if id_kernel==[]:
                return [],[],[]

            line_kernel=(kern.T[ids_[id_kernel],:]@kern[:,:])[0,:]
            top_ids=np.argsort(line_kernel)[::-1][1:ktop+1]
            top_scores=np.sort(line_kernel)[::-1][1:ktop+1]
            return [i for i in all_words[top_ids] if i!='xxx'],top_ids,top_scores
        
        #Else if a multiple word topic is given :
        #Catch only commun co-occurent terms that are ranked in the first top 15 scores of each words'
        else: 
            for w in lw:

                synonyms,ids_retrieved,scores=self.find_synonyms(w,all_words,15)
                for i,syn in enumerate(synonyms):
                    if syn not in lw and syn not in stopwords:
                        dic_ids[syn]=dic_ids.get(syn,[ids_retrieved[i],[]])
                        dic_ids[syn][1].append(w)
                        dic_syns[syn]=dic_syns.get(syn,[0,0]) #count,weights
                        dic_syns[syn][0]+=1
                        dic_syns[syn][1]+=scores[i]



            syns=list(dic_syns.keys())
            scores=[item[1][1] for item in dic_syns.items()]
            counts=[item[1][0] for item in dic_syns.items()]

            final_df=pd.DataFrame(np.c_[syns,counts,scores],columns=['syns','counts','scores'])
            final_df.scores=final_df.scores.astype(float)
            final_df.counts=final_df.counts.astype(int)


            semi_df1=final_df.sort_values(by=['counts','scores'],ascending=False)
            semi_df1=semi_df1[semi_df1.counts>=2].iloc[:ktop]

            final_words=semi_df1.syns.tolist()
            ids_retrieved=[dic_ids[w][0] for w in final_words]
            scores=semi_df1.scores.tolist()
            return final_words,ids_retrieved,scores
    
    
    def query_expansion(self,q,lwords,max_expansion=6,ktop=3):
        
        kern=self.kernel
        ids=self.idx
        words_unsorted=json.load(open(self.save_features_path, mode = 'rb'))
        idx_tosort=np.array(list(words_unsorted.values()))
        idx_sorted=np.argsort(idx_tosort)
        all_words=np.array(list(words_unsorted.keys()))[idx_sorted]
        count=0
        expansion=[""]
        count_unique_words={}
        for topic in lwords:
            words,i,s=self.find_synonyms(topic,all_words,ktop,stopwords=q.split())
            for w in words:
                count_unique_words[w]=count_unique_words.get(w,0)+1
                if count<max_expansion and count_unique_words[w]==1:
                    expansion.append(w)
                    count+=1
                    

        return q+" ".join(expansion),expansion
    
    def predict(self,queries_set,word_topics):
        

        ############  QUERY PROCESING GET THE TFIDF VECTOR OF THE TWO TREATED QUERRIES #################
        size_semi_clusters=int(self.size_cluster/2),int(self.size_cluster)-int(self.size_cluster/2)
        questions=queries_set
        self.querries_raw=queries_set
        
        #First set of querries treated with word importance Infersent ()
        df=generate_querries(self.querries_raw,self.Infersent_wpath,self.Infersent_mpath,self.retriever.stop_words_list)
        querries_set1=df['words_sort'].tolist()
        for i,q in enumerate(querries_set1):
            q=' '.join(list(q))
            q=q.lower()
            q=remove_non_alpha(q)
            q=q.replace('.','')
            querries_set1[i]=q
            
        self.retriever.transform(querries_set1)
        TF_q1=self.retriever.tfidf_matrix.toarray().transpose()
        emb_q1=self.kernel@TF_q1
        q_emb1=np.apply_along_axis(lambda x: x/np.sqrt(np.sum(x**2)), 1,emb_q1.transpose())
        
        #Second set of querries treated with  query expansion based on our semantic kernel fitted on whole corpus
        
        querries_set2=[self.query_expansion(q,word_topics[i],max_expansion=6,ktop=3)[0] for (i,q) in enumerate(questions) ]
        expansions=[self.query_expansion(q,word_topics[i],max_expansion=6,ktop=3)[1] for (i,q) in enumerate(questions) ]
        for i,q in enumerate(querries_set2):
            q=q.lower()
            q=remove_non_alpha(q)
            q=q.replace('.','')
            querries_set2[i]=q
        
        self.retriever.transform(querries_set2)
        TF_q2=self.retriever.tfidf_matrix.toarray().transpose()
        emb_q2=self.kernel@TF_q2
        q_emb2=np.apply_along_axis(lambda x: x/np.sqrt(np.sum(x**2)), 1,emb_q2.transpose())

        expanded_query=querries_set1[0]+' '+' '.join([word for word in querries_set2[0].split() if word not in queries_set[0].lower().split()])+' ?'
        
        print('\n\n')
        print(' Query : \t {} \n Topics : \t {} \n Expansion : \t {} \n'.format(self.querries_raw[0]," | ".join(word_topics[0]),' | '.join(expansions[0])))

        
        self.q_emb=[q_emb1,q_emb2]
        
        ###################### END OF QUERY PROCESSING & BEGINNING OF RETRIEVER LOOP ####################
        
        
        #For the two forms of  querries obtained we compute the following object?
        
        #(1) Each element of rank is a matrix of question size * size documents containing id of best matches 
        #document obtained by descending order. 
        
        #(2) Each element of scores is a matrix of question size * size_documents containing the cosine similarity
        #between docs and querries 
        
        scores=[self.q_emb[i].dot(self.docs_emb.T) for i in range(2)]
        rank=[np.apply_along_axis(lambda x:np.argsort(-x),1,scores[i]) for i in range(2)]
        raw_or_treated=1 #raw text


        all_answers=[]
        all_scores=[]
        all_indices=[]
        all_models=[]
        all_ranks=[]
        all_querries=[]

        self.all_answers_retrieved_raw=[]
        self.all_answers_retrieved=[]
        all_querries_treated=[]
        all_question_ids=[]
        special_answer=np.zeros_like(questions,dtype=int)


        for question in range(len(questions)):
              
            answers=[]
            answers_raw=[]
            dic_answers={}
            rk=0
            
            w_important=[w for w in questions[question].split(" ") if (w==w.upper() and len(w)>1) ]

            for loop in range(2):
              
                count=0
                if w_important !=[] and loop==1 :
                    for w in w_important:
                        try:
                            w=m3_Tfidf(transform_text=False).tokenize(w)[0]
                            w=w.lower()
                            best_id=[i for i in range(rank[loop].shape[1]) if w in self.content_repo[0][rank[loop][question,i]].split(' ')]
                        except:
                            print("an error occured with the important word{}".format(word))

                        if best_id!=[]:
                            id_match=best_id[0]
                            result=self.content_repo[raw_or_treated][rank[loop][question,id_match]]
                            all_question_ids.extend([question])
                            all_indices.extend([rank[loop][question,id_match]])
                            all_scores.extend([scores[loop][question,rank[loop][question,id_match]]])
                            all_answers.extend([result])
                            all_ranks.append(rk+1)
                            all_models.append("harshQA")
                            all_querries.append(self.querries_raw[question])
                            rk+=1
                            special_answer[question]+=1
                            
                for answ in range(size_semi_clusters[loop]):
                    while True:
                        answer_raw=self.content_repo[raw_or_treated][rank[loop][question,count]]
                        answer=self.content_repo[1-raw_or_treated][rank[loop][question,count]]
                        dic_answers[answer]=dic_answers.get(answer,0)+1
                        if dic_answers[answer]==1:
                            break
                        count+=1

                    answers.append(answer)
                    answers_raw.append(answer_raw)


            self.all_answers_retrieved.append(answers)
            self.all_answers_retrieved_raw.append(answers_raw)

                

        #2) Use Bert encoder inside clusters + cosine similarity retrieval
        Qst=questions[question].lower()
        newQst=remove_non_alpha(Qst)
        newQst=newQst.replace('.','')
        all_querries_treated.append(newQst)

        #Prediction with tensorflow app using bert finetuned on re-ranking MsMarco dataset

        o=len(all_querries_treated)
        tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)
        docs=[tuple(array) for array in self.all_answers_retrieved]
        query_id=[str(i) for i in range(o)]
        doc_ids=[tuple(array) for array in list(np.tile([str(i) for i in range(self.size_cluster) ],o).reshape(o,-1))]
        labels=[0 for i in range(self.size_cluster)]
        
        
        print("********* CLUSTER OF {}  DOCS ********** =  ".format(self.size_cluster))
        show=pd.DataFrame([(i[:100]+"...."+i[100:])[:103] for i in self.all_answers_retrieved_raw[0]])
        print(tabulate(show, headers='keys', tablefmt='psql'))
        
        
        
        convert_eval_dataset(tokenizer,
                             self.output_dir+"/tfrecord",
                             self.output_dir +"/tfrecord"+ '/query_doc_ids_' + 'eval' + '.txt',
                             self.max_seq_length,
                             self.max_query_length,
                             all_querries_treated,
                             docs,
                             labels,
                             query_id,
                             doc_ids)
             
        run_msmarco(**self.kwargs_run_msmarco)
        
        #Transform tensorflow results to a nice DataFrame

        ranker=pd.read_csv(self.output_dir+"/bert_checkpoints"+'/msmarco_predictions_eval.tsv', sep='\t',names=['Q_id','Doc_id','Rank','Probs'])
        self.ranker=ranker[ranker.Rank<=5]
        
        for question in range(o):
            rk=0
            ranker_sub=self.ranker[self.ranker['Q_id']==question]
            scores_bert=ranker_sub['Probs'].tolist()
            indices=ranker_sub['Doc_id'].tolist()
            text=[self.all_answers_retrieved_raw[question][i] for i in indices]

            
            for i,c in enumerate(text):
                
                all_ranks.append(rk+1+special_answer[question])
                all_models.append("harshQA")
                all_querries.append(self.querries_raw[question])
                all_question_ids.extend([question])
                rk+=1
            
            all_answers.extend(text)
            all_scores.extend(scores_bert)
            exact_ids1=np.array([i for i in indices if i<size_semi_clusters[0]],dtype=int)
            exact_ids2=np.array([ (i-size_semi_clusters[0]) for i in indices if i>=size_semi_clusters[0]],dtype=int)
            exact_doc_ids_clust1=rank[0][question,:size_semi_clusters[0]][exact_ids1]
            exact_doc_ids_clust2=rank[1][question,:size_semi_clusters[1]][exact_ids2]

            all_indices.extend(exact_doc_ids_clust1)
            all_indices.extend(exact_doc_ids_clust2)

        #all_indices.extend(np.array(range(len(self.content_repo[0])))[rank[question,:self.top_n]])
        final=pd.DataFrame(np.c_[all_question_ids,all_querries,all_models,all_ranks,all_indices,all_answers, all_scores],columns=['Q_ids','Question','Model','Rank','Doc_index','Answer','Score'])
        return final.sort_values(by=['Q_ids','Rank'],ascending=True).drop(columns=['Q_ids'])






