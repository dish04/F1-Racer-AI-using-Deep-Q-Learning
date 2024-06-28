import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Buffer():
    def __init__(self, size, batch_size=32):
        self.buffer = []
        self.size = size
        self.batch_size = batch_size
    
    def add(self, xi, yi):
        self.buffer.append([xi, yi])
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_batch(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return [np.array([x[0] for x in batch]), np.array([x[1] for x in batch])]

class DQLAgent():
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(6,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.1), loss='mse')
        return model
    
    def __init__(self):
        self.train_model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.train_model.get_weights())
        self.memory = Buffer(2048)
        self.gamma = 0.99
        self.alpha = 0.01
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.state = None
        self.action = None
    
    def begin(self, beginState):
        self.state = np.array(beginState).reshape((1, 6))
        print("BEGIN WORKS!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.action = self.train_model.predict(self.state)
    
    def out_action(self):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(['up', 'down', 'left', 'right'])
        else:
            max_index = np.argmax(self.action)
            print(self.action)
            directions = ['up', 'down', 'left', 'right']
            return directions[max_index]
    
    def update_state(self, next_state, reward):
        next_state = np.array(next_state).reshape((1, 6))
        self._train(state=self.state, reward=reward, next_state=next_state)
        self.state = next_state
        self.action = self.train_model.predict(self.state)
    
    def _update_weights(self):
        self.target_model.set_weights(self.train_model.get_weights())
    
    def _train(self, state, reward, next_state):
        target = self.train_model.predict(state)
        if reward == -1:  # Terminal state
            target[0][np.argmax(self.action)] = reward
        else:
            t = self.target_model.predict(next_state)
            target[0][np.argmax(self.action)] = reward + self.gamma * np.amax(t)
        
        self.train_model.fit(state, target, epochs=1, verbose=0)
        self._remember(state, target)
    
    def target_train(self):
        if len(self.memory.buffer) < self.memory.batch_size:
            return
        
        batch = self.memory.get_batch()
        self.target_model.fit(batch[0], batch[1], epochs=1, verbose=0)
        self._update_weights()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _remember(self, state, target):
        self.memory.add(state, target)
    
    def _get_q(self, reward, next_state):
        return reward + self.gamma * np.max(self.target_model.predict(next_state))
