#!/usr/bin/env python
import os.path
import numpy as np
import cPickle as pickle

from sklearn.linear_model import SGDRegressor


def main():
    root = 'run_random'
    r = lambda name: os.path.join(root, name)
    scores = np.fromfile(r('scores'), dtype=np.float32)
    actions = np.fromfile(r('actions'), dtype=np.int8)
    states = np.fromfile(r('states'), dtype=np.float32)\
        .reshape([len(actions), 36])

    rewards = scores.copy()
    prev_s = 0
    for idx, s in enumerate(scores):
        rewards[idx] = s - prev_s
        prev_s = s

    n_actions = 4
    utility_models = [SGDRegressor() for _ in range(n_actions)]
    for action, model in enumerate(utility_models):
        mask = actions == action
        model.fit(states[mask,:], rewards[mask])

    with open('utility_models.pkl', 'wb') as f:
        pickle.dump(utility_models, f)


if __name__ == '__main__':
    main()
