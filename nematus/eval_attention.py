#!/usr/bin/env python3

import argparse
import logging

import tensorflow as tf

from config import load_config_from_json_file
import inference
import model_loader
import rnn_model
from settings import AttnSettings
from transformer import Transformer as TransformerModel
import util
import data_iterator
import pickle


def infer_attn_per_sentence(session, model, text_iterator):
    attn_weights = []
    for xx, yy in text_iterator:
        x, x_mask, y, y_mask = util.prepare_data(xx, yy, 1, maxlen=None)
        feeds = {model.inputs.x: x,
                 model.inputs.x_mask: x_mask,
                 model.inputs.y: y,
                 model.inputs.y_mask: y_mask,
                 model.inputs.training: False}
        attention_weights = session.run(model.decoder.attention_weights, feed_dict=feeds)
        attn_weights.append(attention_weights)
    return attn_weights


def main(settings):
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
    # Start logging.
    level = logging.DEBUG if settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # Create the TensorFlow session.
    tf_config = tf.ConfigProto()
    tf_config.allow_soft_placement = True
    session = tf.Session(config=tf_config)

    # Load config file for each model.
    configs = []
    for model in settings.models:
        config = load_config_from_json_file(model)
        setattr(config, 'reload', model)
        configs.append(config)

    # Create the model graphs and restore their variables.
    logging.debug("Loading models\n")
    models = []
    for i, config in enumerate(configs):
        with tf.variable_scope("model%d" % i) as scope:
            if config.model_type == "transformer":
                model = TransformerModel(config)
            else:
                model = rnn_model.RNNModel(config)
            saver = model_loader.init_or_restore_variables(config, session,
                                                           ensemble_scope=scope)
            models.append(model)

    # TODO Ensembling is currently only supported for RNNs, so if
    # TODO len(models) > 1 then check models are all rnn

    # Translate the source file.
    
    text_iterator = data_iterator.TextIterator(settings.src, settings.tgt, 
                                                source_dicts=configs[0].source_dicts,
                                                target_dict=configs[0].target_dict,
                                                batch_size=settings.minibatch_size,
                                                source_vocab_sizes=configs[0].source_vocab_sizes,
                                                target_vocab_size=configs[0].target_vocab_size,
                                                maxlen=1000000,
                                                model_type='rnn',
                                                keep_data_in_memory=True)
    attn = infer_attn_per_sentence(session, models[0], text_iterator)
    pickle.dump(attn, settings.output)

if __name__ == "__main__":
    # Parse console arguments.
    settings = AttnSettings(from_console_arguments=True)
    main(settings)
