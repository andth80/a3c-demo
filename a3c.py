import numpy as np
import tensorflow as tf
import threading
from collections import namedtuple

import ale

import tf_util
import config


Transition = namedtuple(
    'Transition',
    ['s', 'a', 'r'])

Master = namedtuple(
    'Master',
    ['theta', 'apply_gradients', 'lock'])

Worker = namedtuple(
    'Worker',
    ['theta', 'copy_theta', 'training'])


def create_training_graph(action_count, worker_count):
    """
    Creates the full TF graph including components for the master and n workers.
    :param action_count: number of valid actions the agent can choose between
    :param worker_count: number of workers to create
    :return: master and list of workers
    """
    master = create_training_master(action_count)

    workers = []
    for i in range(worker_count):
        worker = create_training_worker(action_count, i, master.theta)
        workers.append(worker)

    return master, workers


def create_training_master(action_count):
    """
    Creates the bits of the graph which hold operations and placeholders used by the master. Specifically: the master
    copy of the policy and value functions ('theta'); an operation for applying gradients to the variables representing
    theta; and a lock object to be used to protect async reads and writes to theta.
    :param action_count: number of valid actions the agent can choose between
    :return: tuple representing the master section of the graph
    """
    with tf.name_scope('master'):
        master_theta = Theta(action_count)
        apply_gradients = ApplyGradients(master_theta.variables)

    return Master(master_theta, apply_gradients, threading.Lock())


def create_training_worker(action_count, i, master_theta):
    """
    Creates the bits of the graph which hold operations and placeholders used by a single worker. Specifically: a
    local copy of the policy and value functions ('theta'); an operation for copying the values of the master 'theta'
    to this copy of theta; and operations for calculating gradients against the local theta ('training').
    :param action_count: number of valid actions the agent can choose between
    :param i: a number to identify this worker
    :param master_theta: the master theta variables
    :return: tuple representing a single worker section of the graph
    """
    with tf.name_scope('worker{0}'.format(i)):
        worker_theta = Theta(action_count)

        copy_theta = CopyVariables(
            master_theta.variables, worker_theta.variables)

        training = Training(worker_theta)

    return Worker(worker_theta, copy_theta, training)


def play(env, session, theta, recorder=None):
    """
    Plays the game until a terminal state is reached.
    :param env: ALE env to use
    :param session: TF session to use
    :param theta: policy function to use
    :param recorder: if specified, used to record a video of this game
    :return: the total score at the end of the game
    """
    env.reset()
    terminal = False
    total_reward = 0
    s = ale.initialise_s()

    while not terminal:
        a = theta.get_policy_action(session, s)
        (terminal, r, s, screen) = env.act(a)
        total_reward += r

        if recorder is not None:
            recorder.add_frame(screen)

    return total_reward


def start_training(rom_name, session, master, workers, inc_T):
    worker_threads = []
    for (i, worker) in enumerate(workers):
        print(
            'starting thread: {0}/{1}'.format(i + 1, len(workers)), end=('\r'))

        # have to create the ALE environments on the main thread as ALE
        # initialisation is not thread-safe.
        env = ale.Env(rom_name)

        thread = threading.Thread(
            target=_thread_func,
            args=[env, session, master, worker, inc_T])

        worker_threads.append(thread)

    print('')

    for thread in worker_threads:
        thread.start()


def _thread_func(env, session, master, worker, inc_T):
    terminal = True

    while True:

        # reset
        if terminal:
            env.reset()
            terminal = False
            s = ale.initialise_s()

        with master.lock:
            worker.copy_theta.run(session)

        sequence = []
        t = 0

        # act
        while t < config.T_MAX and not terminal:
            a = worker.theta.get_policy_action(session, s)

            (terminal, r, s_prime, _) = env.act(a)
            sequence.append(Transition(s, a, r))

            inc_T.run(session)
            s = s_prime
            t += 1

        # expected future reward from final state
        r_t = 0 if terminal else worker.theta.run_V(session, s_prime)

        # train
        d_theta = worker.training.run(session, sequence, r_t)
        with master.lock:
            master.apply_gradients.run(session, d_theta)


class IncGlobalStep:
    """Holds an operation to increment the global step"""
    def __init__(self):
        global_step = tf.train.create_global_step()
        self.inc_step_op = tf.assign(global_step, global_step + 1)

    def run(self, session):
        session.run(self.inc_step_op)


class Theta:
    """Holds TF variables associated with the policy and value functions and the placeholders and TF operations for
    evaluating them.
    """
    def __init__(self, action_count):
        with tf.name_scope('theta') as scope:
            self.s = tf.placeholder(
                tf.float32, shape=(None, 84, 84, config.M), name='s')

            conv1 = tf_util.conv2d(
                'conv1', self.s / 256, [8, 8], 32, 4, tf.nn.relu)
            conv2 = tf_util.conv2d('conv2', conv1, [4, 4], 64, 2, tf.nn.relu)
            conv3 = tf_util.conv2d('conv3', conv2, [3, 3], 64, 1, tf.nn.relu)

            flattened = tf.reshape(conv3, [-1, 11 * 11 * 64])
            hidden = tf_util.linear('hidden', flattened, 512, tf.nn.relu)

            self.V = tf_util.linear('V', hidden, 1)
            self.pi = tf_util.softmax('pi', hidden, action_count)

            self.variables = tf.trainable_variables(scope=scope)

            self.eval_summary_op = tf.summary.merge(
                tf.get_collection(
                    tf_util.SummaryKeys.EVAL_SUMMARIES, scope=scope))
            self.variable_summary_op = tf.summary.merge(
                tf.get_collection(
                    tf_util.SummaryKeys.VARIABLE_SUMMARIES, scope=scope))

    def run_pi(self, session, s):
        return self._eval(session, self.pi, s)[0]

    def run_V(self, session, s):
        return self._eval(session, self.V, s)[0][0]

    def _eval(self, session, op, s):
        feed_dict = {self.s: np.reshape(s, (1, *s.shape))}
        return session.run(op, feed_dict)

    def get_policy_action(self, session, s):
        pi = self.run_pi(session, s)
        return np.random.choice(pi.shape[0], p=pi)

    def run_eval_summaries(self, session, s):
        return self._eval(session, self.eval_summary_op, s)

    def run_variable_summaries(self, session):
        return session.run(self.variable_summary_op)


class ApplyGradients:
    """Holds operations for applying gradients to a set of variables. The variables to be updated are specified
    in the constructor and then gradients are passed whenever the operation is to be run. Used to apply gradients
    calculated by the workers to the variables held by the master.
    """
    def __init__(self, variables):
        with tf.name_scope('apply_gradients'):
            self.gradients = [
                tf.placeholder(var.dtype, shape=var.shape)
                for var in variables
            ]

            optimiser = tf.train.AdamOptimizer(config.LEARNING_RATE)
            self.apply_gradients_op = optimiser.apply_gradients(
                zip(self.gradients, variables))

    def run(self, session, gradients):
        feed_dict = {
            placeholder: gradients[i]
            for (i, placeholder) in enumerate(self.gradients)
        }

        session.run(self.apply_gradients_op, feed_dict)


class CopyVariables():
    """Holds operations for copying the values of one set of variables to another. From and To variables are specified
    in the constructor and then every time the operation is run it copies the values of the 'from' variables to the
    'to' variables. Used to pass updated values for variables from the master to a worker.
    """
    def __init__(self, from_vars, to_vars):
        with tf.name_scope('copy_variables'):
            self.ops = [
                tf.assign(v, from_vars[i])
                for i, v in enumerate(to_vars)
            ]

    def run(self, session):
        session.run(self.ops)


class Training:
    """Holds placeholders and operations for calculating gradients given a sequence of observations from the
    environment.
    """
    def __init__(self, theta):
        with tf.name_scope('training'):
            self.theta = theta
            self.a = tf.placeholder(tf.int32, name='a')
            self.r = tf.placeholder(tf.float32, name='r')

            advantage = self.r - theta.V
            log_pi = tf.log(theta.pi)
            one_hot_a = tf.one_hot(self.a, theta.pi.shape[1])
            log_pi_a = tf.reduce_sum(log_pi * one_hot_a, axis=1)

            # loss
            entropy = -tf.reduce_sum(log_pi * theta.pi, axis=1)
            pi_loss = -(log_pi_a * tf.stop_gradient(advantage))
            V_loss = tf.square(advantage)

            total_loss = pi_loss + V_loss - (config.BETA * entropy)

            self.gradients_op = tf.gradients(
                tf.reduce_mean(total_loss), theta.variables)

    def run(self, session, sequence, r_t):
        rewards = []
        r_i = r_t
        for step in reversed(sequence):
            r_i = (r_i * config.GAMMA) + step.r
            rewards.insert(0, r_i)

        feed_dict = {
            self.theta.s: [step.s for step in sequence],
            self.a: [step.a for step in sequence],
            self.r: rewards
        }

        return session.run(self.gradients_op, feed_dict)
