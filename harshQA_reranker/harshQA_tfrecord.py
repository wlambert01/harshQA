#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:40:17 2019

@author: williamlambert
"""

"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
"""
import collections
import os
import re
import tensorflow as tf
import time
# local module
import harshQA_reranker.tokenization as tokenization

def write_to_tf_record(writer, tokenizer, max_seq_length,max_query_length,query, docs, labels,
                       ids_file=None, query_id=None, doc_ids=None):
  query = tokenization.convert_to_unicode(query)
  query_token_ids = tokenization.convert_to_bert_input(
      text=query, max_seq_length=max_query_length, tokenizer=tokenizer, 
      add_cls=True)

  query_token_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_token_ids))

  for i, (doc_text, label) in enumerate(zip(docs, labels)):

    doc_token_id = tokenization.convert_to_bert_input(
          text=tokenization.convert_to_unicode(doc_text),
          max_seq_length=max_seq_length - len(query_token_ids),
          tokenizer=tokenizer,
          add_cls=False)

    doc_ids_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=doc_token_id))

    labels_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label]))

    features = tf.train.Features(feature={
        'query_ids': query_token_ids_tf,
        'doc_ids': doc_ids_tf,
        'label': labels_tf,
    })
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())

    if ids_file:
     ids_file.write('\t'.join([query_id, doc_ids[i]]) + '\n')
     

def convert_eval_dataset(tokenizer,output_folder,query_doc_ids_path,max_seq_length,max_query_length, query,docs,labels,query_id,doc_ids):
  set_name='eval'
  print('Converting {} set to tfrecord...'.format(set_name))
  writer = tf.python_io.TFRecordWriter(output_folder + '/dataset_' + set_name + '.tf')
  with open(query_doc_ids_path, 'w') as ids_file:
      for i,q in enumerate(query):
          write_to_tf_record(writer=writer,
                             tokenizer=tokenizer,
                             max_seq_length=max_seq_length,
                             max_query_length=max_query_length,
                             query=q, 
                             docs=docs[i], 
                             labels=labels,
                             ids_file=ids_file,
                             query_id=query_id[i],
                             doc_ids=doc_ids[i])
  print('Done!')
  writer.close()
  return None



