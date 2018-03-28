import os
import tensorflow as tf


class SummaryKeys():
    VARIABLE_SUMMARIES = 'VariableSummaries'
    EVAL_SUMMARIES = 'EvalSummaries'


def create_session(debug=False):
    session = tf.Session()
    if debug:
        session = tf_debug.LocalCLIDebugWrapperSession(session)
        session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    return session


def write_summary(writer, session, summary):
    step = global_step(session)
    writer.add_summary(summary, step)


def write_summary_value(writer, session, name, value):
    step = global_step(session)
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    writer.add_summary(summary, step)


def global_step(session):
    step_tensor = tf.train.get_global_step(session.graph)
    return tf.train.global_step(session, step_tensor)


def save_checkpoint(session, dir):
    step = global_step(session)
    name = 'graph-{0}'.format(step)
    saver = tf.train.Saver()
    checkpoint = saver.save(session, os.path.join(dir, name))
    print('saved checkpoint: ' + checkpoint)

    return checkpoint


def restore_checkpoint(session, checkpoint):
    saver = tf.train.Saver()
    saver.restore(session, checkpoint)
    print('restored checkpoint: ' + checkpoint)


def conv2d(name, x, kernel_shape, filter_count, stride, activation_fn=None):
    with tf.name_scope(name):
        w_init = tf.truncated_normal(
            [*kernel_shape, x.get_shape()[3].value, filter_count],
            stddev=0.1, mean=0.0, dtype=tf.float32)
        kernel = tf.Variable(w_init, name='kernel')
        strides = [1, stride, stride, 1]
        conv = tf.nn.conv2d(x, kernel, strides=strides, padding='SAME')

        b_init = tf.constant(0, shape=[filter_count], dtype=tf.float32)
        b = tf.Variable(b_init, name='bias')

        a = _activation(conv + b, activation_fn)

        tf.summary.histogram(
            'kernel', kernel, collections=[SummaryKeys.VARIABLE_SUMMARIES])
        tf.summary.histogram(
            'biases', b, collections=[SummaryKeys.VARIABLE_SUMMARIES])
#        activation_images = tf.transpose(a, perm=[3, 1, 2, 0])
#        tf.summary.image('activations', activation_images,
#            max_outputs=filter_count, collections=[EVAL_SUMMARIES])

    return a


def linear(name, x, size, activation_fn=None):
    with tf.name_scope(name):
        w_init = tf.truncated_normal(
            [x.get_shape()[1].value, size],
            stddev=0.1,
            mean=0.0,
            dtype=tf.float32)

        w = tf.Variable(w_init, name='weights')

        b_init = tf.constant(0, shape=[size], dtype=tf.float32)
        b = tf.Variable(b_init, name='bias')

        a = _activation(tf.matmul(x, w) + b, activation_fn)

        tf.summary.histogram(
            'weights', w, collections=[SummaryKeys.VARIABLE_SUMMARIES])
        tf.summary.histogram(
            'biases', b, collections=[SummaryKeys.VARIABLE_SUMMARIES])

    return a


def softmax(name, x, size):
    return linear(name, x, size, tf.nn.softmax)


def _activation(a, activation_fn):
    if (activation_fn is not None):
        a = activation_fn(a, name='activation')

    tf.summary.scalar(
        'sparcity', tf.nn.zero_fraction(a),
        collections=[SummaryKeys.EVAL_SUMMARIES])

    tf.summary.histogram(
        'activations', a,
        collections=[SummaryKeys.EVAL_SUMMARIES])

    return a
