#!/usr/bin/env python
from __future__ import print_function
import cPickle as pickle

import numpy as np
import interface as bbox
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.5):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[0]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        indices = np.random.randint(0, len_memory, size=inputs.shape[0])
        t_predictions = model.predict(
            np.array([self.memory[idx][0][0] for idx in indices]))
        tp1_predictions = model.predict(
            np.array([self.memory[idx][0][-1] for idx in indices]))

        for i, idx in enumerate(indices):
            (state_t, action_t, reward_t, _), game_over = self.memory[idx]
            inputs[i] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = t_predictions[i]
            Q_sa = np.max(tp1_predictions[i])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


def main():
    epsilon = .1  # exploration
    num_actions = 4
    input_size = 36
    hidden_size = 24
    activation = 'relu'
    max_memory = 2000
    batch_size = 50
    mini_epoch = 5
    epoch = 10

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=[input_size],
                    activation=activation))
    model.add(Dense(hidden_size, activation=activation))
    model.add(Dense(num_actions))
    model.compile('adam', 'mse')

    # model.load_weights('model.h5')

    # Define environment/game
    bbox.load_level('../levels/train_level.data', verbose=True)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # FIXME
   #states = np.fromfile('run_random/states', dtype=np.float32)\
   #    .reshape([1214494, 36])
   #scaler = preprocessing.StandardScaler()
   #scaler.fit(states)
   #with open('scaler.pkl', 'wb') as f:
   #    scaler = pickle.dump(scaler, f, protocol=-1)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Train
    for e in range(epoch):
        loss = 0.
        bbox.reset_level()
        game_over = False
        # get initial input
        get_state = lambda : scaler.transform(np.array([bbox.get_state()]))[0]
        input_t = get_state()
        score = 0
        step = 0
        report_steps = 100

        while not game_over:
            step += 1
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(np.array([input_tm1]))[0]
                action = np.argmax(q)

            # apply action, get rewards and new state
            game_over = not bbox.do_action(action)
            input_t = get_state()
            new_score = bbox.get_score()
            reward = new_score - score
            score = new_score

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            for _ in range(mini_epoch):
                inputs, targets = exp_replay.get_batch(
                    model, batch_size=batch_size)
                loss += model.train_on_batch(inputs, targets)[0]

            if step % report_steps == 0:
                print('Step {:07d} | Loss {:.4f} | Score {}'.format(
                    step, loss / (report_steps * mini_epoch), score))
                loss = 0.

        print('Epoch {:03d}/{} | Score {}'.format(e, epoch - 1, score))

    # Save trained model weights
    model.save_weights('q_model.h5', overwrite=True)


if __name__ == '__main__':
    main()
