#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
from utils.utils  import hide_warn
warnings.warn=hide_warn
import numpy as np
import tensorflow as tf 
import torch
import torch.nn as nn
from harshQA_reranker.tokenization import* 
import harshQA_reranker.metrics as metrics
import harshQA_reranker.modeling as modeling
import harshQA_reranker.optimization as optimization 
from harshQA_reranker.harshQA_tfrecord import *
from harshQA_reranker.harshQA_bert_builder import *

class msmarco_args():
  def __init__(self,use_tpu,
                tpu_name,
                do_train,
                do_eval,
                max_seq_length,
                max_eval_examples,
                output_dir,
                bert_config_file,
                tpu_zone,
                gcp_project,
                master,
                save_checkpoints_steps,
                eval_batch_size,
                train_batch_size,
                num_warmup_steps,
                num_train_steps,
                learning_rate,
                init_checkpoint,
                num_tpu_cores,
                iterations_per_loop,
                size_cluster,
                msmarco_output):
    return None


def run_msmarco(use_tpu,
                tpu_name,
                do_train,
                do_eval,
                max_seq_length,
                max_eval_examples,
                output_dir,
                bert_config_file,
                tpu_zone,
                gcp_project,
                master,
                save_checkpoints_steps,
                eval_batch_size,
                train_batch_size,
                num_warmup_steps,
                num_train_steps,
                learning_rate,
                init_checkpoint,
                num_tpu_cores,
                iterations_per_loop,
                size_cluster,
                msmarco_output):



    if not do_train and not do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    assert os.path.exists(bert_config_file), "problem with bert_config_file {}".format(bert_config_file)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(output_dir+"/bert_checkpoints")

    tpu_cluster_resolver = None
    if use_tpu and tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=master,
      model_dir=output_dir+"/bert_checkpoints",
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=num_tpu_cores,
          per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=2,
      init_checkpoint=init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      predict_batch_size=eval_batch_size)

    if do_train:
    
        train_input_fn = input_fn_builder(
            dataset_path=output_dir +"/tfrecord"+"/dataset_train.tf",
            seq_length=max_seq_length,
            is_training=True)
        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps)

    if do_eval:
        
        for set_name in ["eval"]:
            
            max_eval_examples_ = None
            if max_eval_examples:
                max_eval_examples_ = max_eval_examples * size_cluster

            eval_input_fn = input_fn_builder(
              dataset_path=output_dir +"/tfrecord" + "/dataset_" + set_name + ".tf",
              seq_length=max_seq_length,
              is_training=False,
              max_eval_examples=max_eval_examples_)

      #tf.logging.info("Getting results ...")

            if msmarco_output:
                
                path_tsv=output_dir +"/bert_checkpoints"+ "/msmarco_predictions_" + set_name + ".tsv"
                assert os.path.exists(path_tsv), "problem with the tsv path {}".format(path_tsv)
                msmarco_file = tf.io.gfile.GFile(
                    path_tsv, "w")
                query_docids_map = []
                with tf.io.gfile.GFile(
                    output_dir +"/tfrecord" + "/query_doc_ids_" + set_name + ".txt") as ref_file:
                    for line in ref_file:
                        query_docids_map.append(line.strip().split("\t"))

            result = estimator.predict(input_fn=eval_input_fn,
                                 yield_single_examples=True)
      #start_time = time.time()
        results = []
      
        example_idx = 0
        total_count = 0
        
        
        for item in result:
            results.append((item["log_probs"], item["label_ids"]))
            #if total_count % 10000 == 0:
              #tf.logging.info("Read {} examples in {} secs".format(
                  #total_count, int(time.time() - start_time)))
            #print("***results***",len(results),results)
        
        if len(results) == size_cluster:
            
            log_probs, labels = zip(*results)
            log_probs = np.stack(log_probs).reshape(-1, 2)
            #print("probs=",np.exp(log_probs))

            labels = np.stack(labels)

            scores = log_probs[:, 1]
            pred_docs = scores.argsort()[::-1]
            #print("scores",scores)
            scores_sorted=np.sort(np.exp(scores))[::-1]
            #print("scores_sorted",scores_sorted,scores_sorted[0])
            #print("pred_docs=",pred_docs)
            #print("logprobs=",log_probs)

            if msmarco_output:
                
                start_idx = example_idx * size_cluster
                end_idx = (example_idx + 1) * size_cluster
                query_ids, doc_ids = zip(*query_docids_map[start_idx:end_idx])
                assert len(set(query_ids)) == 1, "Query ids must be all the same."
                query_id = query_ids[0]
                rk = 1
                for doc_idx in pred_docs:
                    doc_id = doc_ids[doc_idx]
                    # Skip fake docs, as they are only used to ensure that each query
                    # has proper number of docs.
                    if doc_id != "00000000":
                    
                        msmarco_file.write(
                            "\t".join((query_id, doc_id, str(rk), str(scores_sorted[rk-1]) )) + "\n")
                        rk += 1

        example_idx += 1
        results = []

    total_count += 1

    if msmarco_output:
        msmarco_file.close()

