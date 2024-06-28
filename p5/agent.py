import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

class Buffer():
    def __init__(self, size, batch_size=32) -> None:
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
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(400, 400, 3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4))  # No activation function here for Q-values
        model.compile(optimizer=Adam(learning_rate=0.1), loss='mse')
        return model
    
    def __init__(self) -> None:
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
        self.state = np.array(beginState).reshape((1, 400, 400, 3))
        print("BEGGIN WORKS!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.action = self.train_model.predict(self.state)
    
    def out_action(self):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(['up', 'down', 'left', 'right'])
        else:
            max_index = np.argmax(self.action)
            directions = ['up', 'down', 'left', 'right']
            return directions[max_index]
    
    def update_state(self, next, reward):
        next_state = np.array(next).reshape((1, 400, 400, 3))
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

def prep_image(startPos, filePath):
    img = cv2.imread(filePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    startPos = [int(startPos[0]), int(startPos[1])]
    img = img[max(0, startPos[1]*2-200):min(200+startPos[1]*2, img.shape[0]),
              max(0, startPos[0]*2-200):min(200+startPos[0]*2, img.shape[1])]
    if img.shape[0] < 400 or img.shape[1] < 400:
        new_img = np.zeros((200, 200), dtype=img.dtype)
        start_x = (400 - img.shape[0]) // 2
        start_y = (400 - img.shape[1]) // 2
        new_img[start_x:start_x + img.shape[0], start_y:start_y + img.shape[1]] = img
        img = new_img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("prepd.png", img)