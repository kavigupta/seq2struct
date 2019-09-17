
from os.path import expanduser, exists, join
from subprocess import check_call

import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from datetime import datetime


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
LABEL_LIST = 0, 1
LEARNING_RATE = 2e-5
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
MAX_SEQ_LENGTH = 64

LOCAL_DIR = expanduser("~/models")
BERT_MODEL_HUB = "https://storage.googleapis.com/tfhub-modules/google/bert_uncased_L-12_H-768_A-12/1.tar.gz"
LOCAL_BERT = join(LOCAL_DIR, "bert")


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    if not exists(LOCAL_BERT):
        check_call(["mkdir", "-p", LOCAL_BERT])
        check_call(["wget", BERT_MODEL_HUB, "-O", "bert.tar.gz"], cwd=LOCAL_DIR)
        check_call(["tar", "xvzf", "bert.tar.gz", "-C", "bert"], cwd=LOCAL_DIR)
    with tf.Graph().as_default():
        bert_module = hub.Module(LOCAL_BERT)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
            LOCAL_BERT,
            trainable=True)
    bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
    bert_outputs = bert_module(
            inputs=bert_inputs,
            signature="tokens",
            as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                                         num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                        label_ids,
                        predicted_labels)
                auc = tf.metrics.auc(
                        label_ids,
                        predicted_labels)
                recall = tf.metrics.recall(
                        label_ids,
                        predicted_labels)
                precision = tf.metrics.precision(
                        label_ids,
                        predicted_labels)
                true_pos = tf.metrics.true_positives(
                        label_ids,
                        predicted_labels)
                true_neg = tf.metrics.true_negatives(
                        label_ids,
                        predicted_labels)
                false_pos = tf.metrics.false_positives(
                        label_ids,
                        predicted_labels)
                false_neg = tf.metrics.false_negatives(
                        label_ids,
                        predicted_labels)
                return {
                        "eval_accuracy": accuracy,
                        "f1_score": f1_score,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "true_positives": true_pos,
                        "true_negatives": true_neg,
                        "false_positives": false_pos,
                        "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                    loss=loss,
                    train_op=train_op)
            else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                        loss=loss,
                        eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


class MatchModel:
    def __init__(self, model_dir, num_train_epochs, num_train_data, batch_size):
        # Compute # train and warmup steps from batch size
        num_train_steps = int(num_train_data / batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
        self.num_train_steps = num_train_steps
        run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            save_summary_steps=SAVE_SUMMARY_STEPS,
            save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
        model_fn = model_fn_builder(
            num_labels=len(LABEL_LIST),
            learning_rate=LEARNING_RATE,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)
        self.tokenizer = create_tokenizer_from_hub_module()
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": batch_size})
    def are_matches(self, questions, sqls):
        input_examples = [
            run_classifier.InputExample(
                guid="",
                text_a=question,
                text_b=sql,
                label=0
            ) for question, sql in zip(questions, sqls)]
        input_features = run_classifier.convert_examples_to_features(
            input_examples, LABEL_LIST, MAX_SEQ_LENGTH, self.tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(
            features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)
        return list(predictions)
    def _dataframe_to_input_fn(self, dataframe, is_training):
        examples = dataframe.apply(
            lambda x:
                bert.run_classifier.InputExample(
                    guid=None,
                    text_a = x['question'],
                    text_b = x['query'],
                    label = x['is_match']),
            axis = 1)
        features = bert.run_classifier.convert_examples_to_features(
            examples,
            LABEL_LIST,
            MAX_SEQ_LENGTH,
            self.tokenizer)
        return bert.run_classifier.input_fn_builder(
            features=features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=is_training,
            drop_remainder=False)

    def train(self, train_data, test_data):
        print("Converting Data")
        train_input_fn = self._dataframe_to_input_fn(train_data, True)
        test_input_fn = self._dataframe_to_input_fn(test_data, False)
        start = datetime.now()
        print("Starting Training at", start)
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        end = datetime.now()
        print("Done Training at", end)
        result = self.estimator.evaluate(input_fn=test_input_fn, steps=None)
        return result, end - start
