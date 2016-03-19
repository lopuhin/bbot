#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import interface as bbox
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import sgd


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
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
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(
                np.random.randint(0, len_memory, size=inputs.shape[0])):
            (state_t, action_t, reward_t, state_tp1), game_over = \
                self.memory[idx]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


def main():
    epsilon = .1  # exploration
    num_actions = 4
    epoch = 10
    max_memory = 500
    hidden_size = 24
    batch_size = 50
    input_size = 36
    activation = 'sigmoid'

    model = Sequential()
  # model.add(BatchNormalization(input_shape=[input_size,]))
    model.add(Dense(hidden_size, input_shape=[input_size,], activation='sigmoid'))
    model.add(Dense(hidden_size, activation=activation))
    model.add(Dense(num_actions))
   #model.add(Dense(num_actions, input_shape=[input_size,],))
   #model.compile('adam', 'mse')
    model.compile(sgd(lr=.1), 'msle')

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights('model.h5')

    # Define environment/game
    bbox.load_level('../levels/train_level.data', verbose=True)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    for e in range(epoch):
        loss = 0.
        bbox.reset_level()
        game_over = False
        # get initial input
        get_state = lambda : bbox.get_state().reshape((1, -1)) / 10
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
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            game_over = not bbox.do_action(action)
            input_t = get_state()
            new_score = bbox.get_score()
            reward = new_score - score
            score = new_score

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)[0]
            if step % report_steps == 0:
                print('Step {:07d} | Loss {:.4f} | Score {}'.format(
                    step, loss / report_steps, score))
                loss = 0.

        print('Epoch {:03d}/{} | Score {}'.format(e, epoch - 1, score))

    # Save trained model weights
    model.save_weights('q_model.h5', overwrite=True)


if __name__ == '__main__':
    main()
