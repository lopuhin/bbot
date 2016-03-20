#!/usr/bin/env python
from __future__ import print_function
import os.path
import cPickle as pickle

import numpy as np
import interface as bbox


n_features = 36
n_actions = 4


np.random.seed(1)


def run_bbox(verbose=False):
    bbox.load_level("../levels/train_level.data", verbose=True)

    states, actions, scores, rewards = [], [], [], []
    with open('utility_models.pkl', 'rb') as f:
        utility_models = pickle.load(f)

    step = 0
    has_next = 1
    while has_next:
        step += 1
        state = bbox.get_state()
        action = np.random.choice(n_actions)
        utilities = [m.predict([state]) for m in utility_models]
        action = np.argmax(utilities)
        # Do action and bookkeeping
        has_next = bbox.do_action(action)
        states.append(np.array(state))
        actions.append(action)
        score = bbox.get_score()
        rewards.append(score if not scores else (score - scores[-1]))
        scores.append(score)
        if verbose and step % 10000 == 0:
            print(step, score)

    i = 1
    get_outdir = 'run_{}'.format
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

    bbox.finish(verbose=True)


if __name__ == "__main__":
    run_bbox(verbose=True)
