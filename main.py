import sys
import os
from time import sleep
from datetime import datetime

import numpy as np
import tensorflow as tf

import ale

import tf_util
import timer
import videorecorder
import a3c


def get_commandline_args():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage:', sys.argv[0], 'rom_file [checkpoint]')
        exit()

    rom_name = sys.argv[1]

    if len(sys.argv) == 3:
        checkpoint = sys.argv[2]
    else:
        checkpoint = None

    return rom_name, checkpoint


def create_output_path():
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join('output', now)


def create_and_init_graph(session, action_count, checkpoint=None):
    # build graph
    inc_T = a3c.IncGlobalStep()
    master, workers = a3c.create_training_graph(action_count, 4)

    if checkpoint is not None:
        tf_util.restore_checkpoint(session, checkpoint)
    else:
        init = tf.global_variables_initializer()
        session.run(init)

    return master, workers, inc_T


def record_episode(env, session, theta, recorder, name):
    recorder.start_recording(name)
    a3c.play(env, session, theta, recorder)
    recorder.stop_recording()


def play_episodes(env, session, theta, count):
    scores = []
    for i in range(count):
        print('playing: {0}/{1}'.format(i + 1, count), end='\r')
        score = a3c.play(env, session, theta)
        scores.append(score)
    print('')

    return np.mean(scores)


def record_training_rate(session, timer, writer=None):
    rate = timer.split(tf_util.global_step(session))
    if writer:
        tf_util.write_summary_value(writer, session, 'steps per sec', rate)

    print('completed epoch, rate: {0:.2f} steps/sec'.format(rate))


def write_theta_summaries(writer, session, theta, s):
    tf_util.write_summary(
        writer,
        session,
        theta.run_eval_summaries(session, s))

    tf_util.write_summary(
        writer,
        session,
        theta.run_variable_summaries(session))


def main(rom_name, checkpoint):

    output_dir = create_output_path()

    recorder = videorecorder.VideoRecorder(os.path.join(output_dir, 'videos'))

    # start ale
    env = ale.Env(rom_name)

    # setup tensorflow
    session = tf_util.create_session(debug=False)
    master, workers, inc_T = create_and_init_graph(
        session,
        env.get_legal_action_count(),
        checkpoint)

    theta = master.theta

    writer = tf.summary.FileWriter(os.path.join(output_dir, 'summaries'))
    writer.add_graph(session.graph)

    training_timer = timer.Timer()
    training_timer.split(tf_util.global_step(session))

    # start training
    a3c.start_training(rom_name, session, master, workers, inc_T)

    # start evaluation loop
    n = 0
    while (True):
        # wait for some more training to be done
        sleep(60)

        write_theta_summaries(writer, session, theta, env.get_s())

        # Every so often run some episodes to see how we're getting on:
        if n % 10 == 0:
            # 1. run a series of episodes and record the average score
            average = play_episodes(env, session, theta, 20)
            tf_util.write_summary_value(writer, session, 'avg score', average)

            # 2. video a single episode
            episode_name = 'episode{0}'.format(n)
            record_episode(env, session, theta, recorder, episode_name)

            # 3. measure how fast we're training
            record_training_rate(session, training_timer, writer)

            # 4. finally, save a checkpoint
            tf_util.save_checkpoint(
                session, os.path.join(output_dir, 'checkpoints'))

        n += 1


if __name__ == '__main__':
    rom_name, checkpoint = get_commandline_args()
    main(rom_name, checkpoint)
