import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam

from collections import  deque
import random


class DQNAgent:
    def __init__(self, max_replay_memory_size=100_000, min_replay_memory_size=1_000,
                 minibatch_size=64, discount=0.99):

        self.min_replay_memory_size = min_replay_memory_size
        self.minibatch_size = minibatch_size
        self.discount = discount

        # Main model
        self.model = self.create_model()
        # Target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=max_replay_memory_size)

        self.target_update_counter = 0


    def create_model(self):

        coin_price_input = Input(shape=(60,5), name='coin_price_input')
        coin_volume_input = Input(shape=(60,5), name='coin_volume_input')
        inventory_input = Input(shape=(5), name='inventory_input')

        cp_lstm_layer = LSTM(128)(coin_price_input)
        cp_batch_norm_layer = BatchNormalization()(cp_lstm_layer)

        cv_lstm_layer = LSTM(128)(coin_volume_input)
        cv_batch_norm_layer = BatchNormalization()(cv_lstm_layer)


        i_fc_layer = Dense(8,activation='relu')(inventory_input)
        i_batch_norm_layer = BatchNormalization()(i_fc_layer)

        concat_feat  = Concatenate(axis=-1)([cp_batch_norm_layer,
                                             cv_batch_norm_layer,
                                             i_batch_norm_layer])

        fc_layer_1 = Dense(64, activation='relu')(concat_feat)
        dropout_layer_1 = Dropout(0.2)(fc_layer_1)

        fc_layer_2 = Dense(64, activation='relu') (dropout_layer_1)
        dropout_layer_2 = Dropout(0.2)(fc_layer_2)

        outputs = Dense(5, activation = 'linear')(dropout_layer_2)

        model = Model(inputs=[coin_price_input, coin_volume_input,
                      inventory_input], outputs=outputs)

        model.compile(loss='mse', optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = [transition[0] for transition in minibatch]
        current_states1 = np.array([state[0] for state in current_states])
        current_states2 = np.array([state[1] for state in current_states])
        current_states3 = np.array([state[2] for state in current_states])

        current_qs_list = self.model.predict(x=[current_states1,current_states2,
                                                current_states3])

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = [transition[3] for transition in minibatch]
        new_current_states1=np.array([state[0] for state in new_current_states])
        new_current_states2=np.array([state[1] for state in new_current_states])
        new_current_states3=np.array([state[2] for state in new_current_states])

        future_qs_list = self.target_model.predict(x=[new_current_states1,
                                                      new_current_states2,
                                                      new_current_states3])

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            #if not done or done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.discount * max_future_q
            #else:
            #    new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        x1 = np.array([state[0] for state in X])
        x2 = np.array([state[1] for state in X])
        x3 = np.array([state[2] for state in X])

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(x=[x1, x2, x3], y= np.array(y), batch_size=self.minibatch_size,
                       verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):

        x = [np.array(state[0]).reshape(-1,*state[0].shape),
             np.array(state[1]).reshape(-1, *state[1].shape),
             np.array(state[2]).reshape(-1, *state[2].shape)]

        return self.model.predict(x)[0]