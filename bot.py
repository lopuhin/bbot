#!/usr/bin/env python
from __future__ import print_function
import os.path
import random

import numpy as np
import interface as bbox


def get_action_by_state(state, verbose=0):
    if verbose:
        for i in xrange(n_features):
            print "state[%d] = %f" %  (i, state[i])

        print "score = {}, time = {}".format(bbox.get_score(), bbox.get_time())

    return random.randint(0, 3)


n_features = n_actions = -1


def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()


def run_bbox(verbose=False):
    has_next = 1

    prepare_bbox()

    states, actions, scores = [], [], []

    while has_next:
        state = bbox.get_state()
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)

        states.append(np.array(state))
        actions.append(action)
        scores.append(bbox.get_score())

    get_outdir = lambda i: 'run_{}'.format(i)
    i = 1
    outdir = get_outdir(i)
    while os.path.exists(outdir):
        i += 1
        outdir = get_outdir(i)
    os.mkdir(outdir)
    print('saving to {}'.format(outdir))
    scores = np.array(scores, dtype=np.float32)
    scores.tofile(os.path.join(outdir, 'scores'))
    actions = np.array(actions, dtype=np.int8)
    actions.tofile(os.path.join(outdir, 'actions'))
    states = np.array(states, dtype=np.float32)
    states.tofile(os.path.join(outdir, 'states'))

    bbox.finish(verbose=1)


if __name__ == "__main__":
    run_bbox(verbose=0)
