#!/usr/bin/env python
from __future__ import print_function
import os.path

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.utils.validation import NotFittedError
import interface as bbox


n_features = 36
n_actions = 4


np.random.seed(1)


def run_bbox(verbose=False):
    bbox.load_level("../levels/train_level.data", verbose=True)

    states, actions, scores, rewards = [], [], [], []
    utility_models = [SGDRegressor(
        learning_rate='constant',
        #penalty='elasticnet',
        ) for _ in range(n_actions)]
    zero_utilities = np.zeros([n_actions])

    n_past_act = 1
    n_past_st = 0  # in addition to current
    discount = 0.9
    random_steps = 10000

    step = 0
    has_next = 1
    while has_next:
        step += 1
        state = bbox.get_state()
        utilities = zero_utilities
        # Choose action using current utility_models
        if step > random_steps:
            clf_state = np.concatenate(states[-n_past_st:] + [state]) \
                        if n_past_st else state
            try:
                utilities = np.array([
                    m.predict([clf_state])[0] for m in utility_models])
            except NotFittedError:
                pass
       #utilities -= utilities.min()
       #p = None if np.isclose(utilities, 0).all() else \
       #    utilities / utilities.sum()
        if np.random.rand() < 0.1 or step <= random_steps:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(utilities)
        # Do action and bookkeeping
        has_next = bbox.do_action(action)
        states.append(np.array(state))
        actions.append(action)
        score = bbox.get_score()
        rewards.append(score if not scores else (score - scores[-1]))
        scores.append(score)
        # Train classifiers
        if len(rewards) >= n_past_act + n_past_st:
            total_reward = sum(r * np.power(discount, i)
                               for i, r in enumerate(rewards[-n_past_act:]))
            if n_past_act == 1:
                clf_state = np.concatenate(states[-(n_past_act + n_past_st):])
            else:
                clf_state = np.concatenate(
                    states[-(n_past_act + n_past_st) : -n_past_act + 1])
            utility_models[actions[-n_past_act]].partial_fit(
                [clf_state], [total_reward])
        if verbose and step % 1000 == 0:
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
