import logging
import sys
import time

import numpy
import tensorflow as tf

import exception
import util

def beam_search(models, session, x_in, x_mask_in, beam_size, target_lang):
    # x_in is a numpy array with shape (factors, seqLen, batch)
    # x_mask is a numpy array with shape (seqLen, batch)
    x_in = numpy.repeat(x_in, repeats=beam_size, axis=-1)
    x_mask_in = numpy.repeat(x_mask_in, repeats=beam_size, axis=-1)
    feeds = {}
    for model in models:
        # change init_state, context, context_in_attention_layer
        feeds[model.inputs.x] = x_in
        feeds[model.inputs.x_mask] = x_mask_in
        feeds[model.inputs.target_lang_id] = target_lang
    beam_ys, parents, cost = construct_beam_search_functions(models, beam_size)
    beam_ys_out, parents_out, cost_out = session.run(
                                                    [beam_ys, parents, cost],
                                                    feed_dict=feeds)
    return reconstruct_hypotheses(beam_ys_out, parents_out, cost_out, beam_size)

def construct_beam_search_functions(models, beam_size):
    """
    Strategy:
        compute the log_probs - same as with sampling
        for sentences that are ended set log_prob(<eos>)=0, log_prob(not eos)=-inf
        add previous cost to log_probs
        run top k -> (idxs, values)
        use values as new costs
        divide idxs by num_classes to get state_idxs
        use gather to get new states
        take the remainder of idxs after num_classes to get new_predicted words
    """

    # Get some parameter settings.  For ensembling, some parameters are required
    # to be consistent across all models but others are not.  In the former
    # case, we assume that consistency has already been checked.  For the
    # parameters that are allowed to vary across models, the first model's
    # settings take precedence.
    decoder = models[0].decoder
    batch_size = tf.shape(decoder.init_state)[0]
    embedding_size = decoder.embedding_size
    translation_maxlen = decoder.translation_maxlen
    target_vocab_size = decoder.target_vocab_size
    high_depth = 0 if decoder.high_gru_stack == None \
                   else len(decoder.high_gru_stack.grus)

    # Initialize loop variables
    i = tf.constant(0)
    init_ys = -tf.ones(dtype=tf.int32, shape=[batch_size])
    init_embs = [tf.zeros(dtype=tf.float32, shape=[batch_size,embedding_size])] * len(models)

    f_min = numpy.finfo(numpy.float32).min
    init_cost = [0.] + [f_min]*(beam_size-1) # to force first top k are from first hypo only
    init_cost = tf.constant(init_cost, dtype=tf.float32)
    init_cost = tf.tile(init_cost, multiples=[batch_size/beam_size])
    ys_array = tf.TensorArray(
                dtype=tf.int32,
                size=translation_maxlen,
                clear_after_read=True,
                name='y_sampled_array')
    p_array = tf.TensorArray(
                dtype=tf.int32,
                size=translation_maxlen,
                clear_after_read=True,
                name='parent_idx_array')
    init_base_states = [m.decoder.init_state for m in models]
    init_high_states = [[m.decoder.init_state] * high_depth for m in models]
    init_loop_vars = [i, init_base_states, init_high_states, init_ys, init_embs,
                      init_cost, ys_array, p_array]

    # Prepare cost matrix for completed sentences -> Prob(EOS) = 1 and Prob(x) = 0
    eos_log_probs = tf.constant(
                        [[0.] + ([f_min]*(target_vocab_size - 1))],
                        dtype=tf.float32)
    eos_log_probs = tf.tile(eos_log_probs, multiples=[batch_size,1])

    def cond(i, prev_base_states, prev_high_states, prev_ys, prev_embs, cost, ys_array, p_array):
        return tf.logical_and(
                tf.less(i, translation_maxlen),
                tf.reduce_any(tf.not_equal(prev_ys, 0)))

    def body(i, prev_base_states, prev_high_states, prev_ys, prev_embs, cost, ys_array, p_array):
        # get predictions from all models and sum the log probs
        sum_log_probs = None
        base_states = [None] * len(models)
        high_states = [None] * len(models)
        for j in range(len(models)):
            d = models[j].decoder
            states1 = d.grustep1.forward(prev_base_states[j], prev_embs[j])
            att_ctx = d.attstep.forward(states1)
            base_states[j] = d.grustep2.forward(states1, att_ctx)
            if d.high_gru_stack == None:
                stack_output = base_states[j]
                high_states[j] = []
            else:
                if d.high_gru_stack.context_state_size == 0:
                    stack_output, high_states[j] = d.high_gru_stack.forward_single(
                        prev_high_states[j], base_states[j])
                else:
                    stack_output, high_states[j] = d.high_gru_stack.forward_single(
                        prev_high_states[j], base_states[j], context=att_ctx)
            logits = d.predictor.get_logits(prev_embs[j], stack_output,
                                            att_ctx, multi_step=False)
            log_probs = tf.nn.log_softmax(logits) # shape (batch, vocab_size)
            if sum_log_probs == None:
                sum_log_probs = log_probs
            else:
                sum_log_probs += log_probs

        # set cost of EOS to zero for completed sentences so that they are in top k
        # Need to make sure only EOS is selected because a completed sentence might
        # kill ongoing sentences
        sum_log_probs = tf.where(tf.equal(prev_ys, 0), eos_log_probs, sum_log_probs)

        all_costs = sum_log_probs + tf.expand_dims(cost, axis=1) # TODO: you might be getting NaNs here since -inf is in log_probs

        all_costs = tf.reshape(all_costs,
                               shape=[-1, target_vocab_size * beam_size])
        values, indices = tf.nn.top_k(all_costs, k=beam_size) #the sorted option is by default True, is this needed? 
        new_cost = tf.reshape(values, shape=[batch_size])
        offsets = tf.range(
                    start = 0,
                    delta = beam_size,
                    limit = batch_size,
                    dtype=tf.int32)
        offsets = tf.expand_dims(offsets, axis=1)
        survivor_idxs = (indices/target_vocab_size) + offsets
        new_ys = indices % target_vocab_size
        survivor_idxs = tf.reshape(survivor_idxs, shape=[batch_size])
        new_ys = tf.reshape(new_ys, shape=[batch_size])
        new_embs = [m.decoder.y_emb_layer.forward(new_ys, m.inputs.target_lang_id) for m in models]
        for new_emb in new_embs:
            # We need to specify the shape of new_emb to avoid a shape invariant
            # failure - it seems that TensorFlow can't fully infer the shape
            # under certain circumstances (specifically, if we're using tied
            # encoder-decoder embeddings and y_emb_layer.forward involves a
            # slice).
            new_emb.set_shape([None, embedding_size])
        new_base_states = [tf.gather(s, indices=survivor_idxs) for s in base_states]
        new_high_states = [[tf.gather(s, indices=survivor_idxs) for s in states] for states in high_states]
        new_cost = tf.where(tf.equal(new_ys, 0), tf.abs(new_cost), new_cost)

        ys_array = ys_array.write(i, value=new_ys)
        p_array = p_array.write(i, value=survivor_idxs)

        return i+1, new_base_states, new_high_states, new_ys, new_embs, new_cost, ys_array, p_array


    final_loop_vars = tf.while_loop(
                        cond=cond,
                        body=body,
                        loop_vars=init_loop_vars,
                        back_prop=False)
    i, _, _, _, _, cost, ys_array, p_array = final_loop_vars

    indices = tf.range(0, i)
    sampled_ys = ys_array.gather(indices)
    parents = p_array.gather(indices)
    cost = tf.abs(cost) #to get negative-log-likelihood
    return sampled_ys, parents, cost


def reconstruct_hypotheses(ys, parents, cost, beam_size):
        #ys.shape = parents.shape = (seqLen, beam_size x batch_size) 
        # output: hypothesis list with shape (batch_size, beam_size, (sequence, cost))

        def reconstruct_single(ys, parents, hypoId, hypo, pos):
            if pos < 0:
                hypo.reverse()
                return hypo
            else:
                hypo.append(ys[pos, hypoId])
                hypoId = parents[pos, hypoId]
                return reconstruct_single(ys, parents, hypoId, hypo, pos - 1)

        hypotheses = []
        batch_size = ys.shape[1] / beam_size
        pos = ys.shape[0] - 1
        for batch in range(batch_size):
            hypotheses.append([])
            for beam in range(beam_size):
                i = batch*beam_size + beam
                hypo = reconstruct_single(ys, parents, i, [], pos)
                hypo = numpy.trim_zeros(hypo, trim='b') # b for back
                hypo.append(0)
                hypotheses[batch].append((hypo, cost[i]))
        return hypotheses


def translate_file(input_file, output_file, source_lang, target_lang, session,
                   models, config, beam_size=12, nbest=False, minibatch_size=80,
                   maxibatch_size=20, normalization_alpha=1.0):
    """Translates a source file using a translation model (or ensemble).

    Args:
        input_file: file object from which source sentences will be read.
        output_file: file object to which translations will be written.
        source_lang: ...
        target_lang: ...
        session: TensorFlow session.
        models: list of model objects to use for ensemble beam search.
        config: model config (must be valid for all models).
        beam_size: beam width.
        nbest: if True, produce n-best output with scores; otherwise 1-best.
        minibatch_size: minibatch size in sentences.
        maxibatch_size: number of minibatches to read and sort, pre-translation.
        normalization_alpha: alpha parameter for length normalization.
    """

    def normalize(sent, cost):
        return (sent, cost / (len(sent) ** normalization_alpha))

    def translate_maxibatch(maxibatch, num_to_target, num_prev_translated):
        """Translates an individual maxibatch.

        Args:
            maxibatch: a list of sentences.
            num_to_target: dictionary mapping target vocabulary IDs to strings.
            num_prev_translated: the number of previously translated sentences.
        """

        # Sort the maxibatch by length and split into minibatches.
        try:
            minibatches, idxs = util.read_all_lines(config, maxibatch,
                                                    minibatch_size, source_lang)
        except exception.Error as x:
            logging.error(x.msg)
            sys.exit(1)

        # Translate the minibatches and store the resulting beam (i.e.
        # translations and scores) for each sentence.
        beams = []
        for x in minibatches:
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = util.prepare_data(x, y_dummy, config.factors,
                                                maxlen=None)
            sample = beam_search(models, session, x, x_mask, beam_size,
                                 target_lang)
            beams.extend(sample)
            num_translated = num_prev_translated + len(beams)
            logging.info('Translated {} sents'.format(num_translated))

        # Put beams into the same order as the input maxibatch.
        tmp = numpy.array(beams, dtype=numpy.object)
        ordered_beams = tmp[idxs.argsort()]

        # Write the translations to the output file.
        for i, beam in enumerate(ordered_beams):
            if normalization_alpha:
                beam = map(lambda (sent, cost): normalize(sent, cost), beam)
            beam = sorted(beam, key=lambda (sent, cost): cost)
            if nbest:
                num = num_prev_translated + i
                for sent, cost in beam:
                    translation = util.seq2words(sent,
                                                 num_to_target[target_lang])
                    line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                    output_file.write(line)
            else:
                best_hypo, cost = beam[0]
                line = util.seq2words(best_hypo,
                                      num_to_target[target_lang]) + '\n'
                output_file.write(line)

    _, _, _, num_to_target = util.load_dictionaries(config)

    logging.info("NOTE: Length of translations is capped to {}".format(
        config.translation_maxlen))

    start_time = time.time()

    num_translated = 0
    maxibatch = []
    while True:
        line = input_file.readline()
        if line == "":
            if len(maxibatch) > 0:
                translate_maxibatch(maxibatch, num_to_target, num_translated)
                num_translated += len(maxibatch)
            break
        maxibatch.append(line)
        if len(maxibatch) == (maxibatch_size * minibatch_size):
            translate_maxibatch(maxibatch, num_to_target, num_translated)
            num_translated += len(maxibatch)
            maxibatch = []

    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(
        num_translated, duration, num_translated/duration))
