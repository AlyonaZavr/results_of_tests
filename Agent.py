from collections import deque
import random
import numpy as np
from keras import Sequential

from keras.layers import Dense
from keras.optimizers import Adam


class DQN_agent:
    def __init__(self, params):
        self.eter = 0
        self.env_info = {"state_space": params["state_space"]}
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.layer_sizes = params['layer_sizes']
        self.memory = deque(maxlen=2500)
        self.action_space = 8
        self.state_space = 16
        self.target_net_1 = []
        self.target_net_2 = []
        if params["name1"] and params["name2"]:
            self.load_model(params.get("name1"), params.get("name2"))
        else:
            self.model1 = self.build_model()
            self.model2 = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.layer_sizes[0], input_shape=(self.state_space,), activation='relu'))
        for i in range(1, len(self.layer_sizes)):
            model.add(Dense(self.layer_sizes[i], activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model1.predict([state])
        return np.argmax(act_values[0])

        # ////////////////////////////////////////////////////////////////
        # Overestimation Bias

    def replay_DQNB(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        predicts = self.model1.predict_on_batch(next_states)
        ind = []
        for i in predicts:
            ind.append(np.argmax(i))

        predicts = self.model2.predict_on_batch(next_states)
        targets1 = []
        for i, j in zip(predicts, range(len(ind))):
            targets1.append(self.gamma * i[ind[j]] + rewards[j])
        targets1 = np.array(targets1)
        targets_full1 = self.model1.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full1[[ind], [actions]] = targets1

        predicts = self.model2.predict_on_batch(next_states)
        ind = []
        for i in predicts:
            ind.append(np.argmax(i))

        predicts = self.model1.predict_on_batch(next_states)
        targets2 = []
        for i, j in zip(predicts, range(len(ind))):
            targets2.append(self.gamma * i[ind[j]] + rewards[j])

        targets2 = np.array(targets1)
        targets_full2 = self.model2.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full2[[ind], [actions]] = targets2

        self.target_net_1 = targets_full1
        self.target_net_2 = targets_full2

        self.model1.fit(states, targets_full1, epochs=1,
                        verbose=0)  # для упрощения работы подгружается сразу вся информация
        self.model2.fit(states, targets_full2, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        # ////////////////////////////////////////////////////////////////
        # Twin DQN

    def replay_TDQN(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        predicts1 = self.model1.predict_on_batch(next_states)
        ind = []
        for i in predicts1:
            ind.append(np.argmax(i))

        predicts2 = self.model2.predict_on_batch(next_states)
        targets2 = []
        for i, k, j in zip(predicts1, predicts2, range(len(ind))):
            if i[ind[j]] < k[ind[j]]:
                minimum = i[ind[j]]
            else:
                minimum = k[ind[j]]
            targets2.append(self.gamma * minimum + rewards[j])

        targets2 = np.array(targets2)
        targets_full2 = self.model2.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full2[[ind], [actions]] = targets2

        #  _______________________________________________________

        predicts2 = self.model2.predict_on_batch(next_states)
        ind = []
        for i in predicts2:
            ind.append(np.argmax(i))

        predicts1 = self.model1.predict_on_batch(next_states)
        targets1 = []
        for i, k, j in zip(predicts1, predicts2, range(len(ind))):
            if i[ind[j]] < k[ind[j]]:
                minimum = i[ind[j]]
            else:
                minimum = k[ind[j]]
            targets1.append(self.gamma * minimum + rewards[j])

        targets1 = np.array(targets1)
        targets_full1 = self.model1.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full1[[ind], [actions]] = targets1

        self.model1.fit(states, targets_full1, epochs=1, verbose=0)
        self.model2.fit(states, targets_full2, epochs=1, verbose=0)

        self.target_net_1 = targets_full1
        self.target_net_2 = targets_full2

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        # ////////////////////////////////////////////////////////////////

    def save_target_net(self):
        import pandas as pd
        for i in range(len(self.target_net_1)):
            for j in range(len(self.target_net_1[i])):
                if self.target_net_1[i][j] < 0.001:
                    self.target_net_1[i][j] = 0

        for i in range(len(self.target_net_2)):
            for j in range(len(self.target_net_2[i])):
                if self.target_net_2[i][j] < 0.001:
                    self.target_net_2[i][j] = 0

        df1 = pd.DataFrame(self.target_net_1)
        df2 = pd.DataFrame(self.target_net_2)

        print(df1)

        df1.to_csv("target_net_1.csv")
        df2.to_csv("target_net_2.csv")
        print("Target nets saved")

    def save_model(self, data):
        from os import mkdir
        from zlib import crc32

        try:
            mkdir("../models/")
        except Exception:
            pass

        tmp = ""
        for i in data:
            tmp += str(i[0] + i[1])

        code = crc32(bytes(tmp, "utf-8"))
        name = str(code) + "model1.keras"

        self.model1.save(name)

        name = str(code) + "model2.keras"
        self.model2.save(name)
        print("model saved")

    def load_model(self, name1, name2):
        from tensorflow import keras
        self.model1 = keras.models.load_model(name1)
        self.model2 = keras.models.load_model(name2)
        print("loaded")
