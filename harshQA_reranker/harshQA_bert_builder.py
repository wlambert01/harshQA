#!/usr/bin/env python
# coding: utf-8


import warnings
from utils.utils  import hide_warn
warnings.warn=hide_warn
from harshQA_reranker.tokenization import* 
import harshQA_reranker.metrics as metrics
import harshQA_reranker.modeling as modeling
import harshQA_reranker.optimization as optimization 
from harshQA_reranker.harshQA_tfrecord import *
import torch
import torch.nn as nn
import tensorflow as tf 


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
      # I.e., 0.1 dropout
          output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, log_probs) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = []

        #Initializes current variables with tensors loaded from given checkpoint.
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              predictions={
                  "log_probs": log_probs,
                  "label_ids": label_ids,
              },
              scaffold_fn=scaffold_fn)

        else:
            raise ValueError(
              "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec
    return model_fn

def input_fn_builder(dataset_path, seq_length, is_training,
                     max_eval_examples=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
    

        batch_size = params["batch_size"]
        output_buffer_size = batch_size * 1000

        def extract_fn(data_record):
            features = {
              "query_ids": tf.FixedLenSequenceFeature(
                  [], tf.int64, allow_missing=True),
              "doc_ids": tf.FixedLenSequenceFeature(
                  [], tf.int64, allow_missing=True),
              "label": tf.FixedLenFeature([], tf.int64),
            }
            sample = tf.parse_single_example(data_record, features)

            query_ids = tf.cast(sample["query_ids"], tf.int32)
            doc_ids = tf.cast(sample["doc_ids"], tf.int32)
            label_ids = tf.cast(sample["label"], tf.int32)
            input_ids = tf.concat((query_ids, doc_ids), 0)

            query_segment_id = tf.zeros_like(query_ids)
            doc_segment_id = tf.ones_like(doc_ids)
            segment_ids = tf.concat((query_segment_id, doc_segment_id), 0)

            input_mask = tf.ones_like(input_ids)

            features = {
              "input_ids": input_ids,
              "segment_ids": segment_ids,
              "input_mask": input_mask,
              "label_ids": label_ids,
            }
            return features

        dataset = tf.data.TFRecordDataset([dataset_path])
        dataset = dataset.map(
            extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=1000)
        else:
            if max_eval_examples:
            # Use at most this number of examples (debugging only).
                dataset = dataset.take(max_eval_examples)
            # pass

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "input_ids": [seq_length],
                "segment_ids": [seq_length],
                "input_mask": [seq_length],
                "label_ids": [],
            },
            padding_values={
                "input_ids": 0,
                "segment_ids": 0,
                "input_mask": 0,
                "label_ids": 0,
            },
            drop_remainder=True)

        return dataset
    return input_fn

