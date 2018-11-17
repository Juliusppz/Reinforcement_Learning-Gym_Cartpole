import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

episodes = 2000

discount = 0.95 
experimentstart = 1.0 
experimentreduction = 0.003 
batchsize = 32
punishment = -10

env = gym.make('CartPole-v1')
statedim = env.observation_space.shape[0]
actiondim = env.action_space.n

nn = Sequential()
nn.add(Dense(30, input_dim=statedim, activation='relu'))
nn.add(Dense(30, activation='relu'))
nn.add(Dense(actiondim, activation='linear'))
nn.compile(loss='mse', optimizer=Adam(lr=0.001))

minfails = 10  
history = deque(maxlen=3000)
failhistory = deque(maxlen=minfails)
lastrewards = deque(maxlen=100)

done = False
solved = False

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, statedim])
    for tick in range(500):
        # env.render() # omit the visualization to speed up simulation
        if np.random.rand() <= experimentstart:
            action = random.randrange(actiondim) 
        else:
            actionvalues = nn.predict(state) 
            action = np.argmax(actionvalues[0])  

        nextstate, reward, done, _ = env.step(action)
        reward = reward if not done else punishment
        nextstate = np.reshape(nextstate, [1, statedim])
        history.append((state, action, reward, nextstate, done))
        state = nextstate
        if done:
            failhistory.append((state, action, reward, nextstate, done))
            lastrewards.append(tick)
            # print("episode " + str(episode) + "  average rewards: " + str(sum(lastrewards) / len(lastrewards)) +
            #       "  current rewards: " + str(tick))
            if sum(lastrewards) / len(lastrewards) > 195.0 and not solved:
                print("solved after " + str(episode) + " episodes")
                solved = True
            break
    if len(history) > batchsize:
        if sum(np.equal(np.array(history)[:, 2], punishment)) < minfails and episode > 100:
            for state, action, reward, nextstate, done in failhistory:
                history.append(
                    (state, action, reward, nextstate, done))  # if there are too few failures in history, add some
        batch = random.sample(history, batchsize)
        for state, action, reward, nextstate, done in batch:
            if done:
                target = reward 
            else:
                target = (reward + discount * np.amax(nn.predict(nextstate)[0])) 
            targets = nn.predict(state) 
            targets[0][action] = target
            nn.fit(state, targets, epochs=1, verbose=0)
    experimentstart *= (1 - experimentreduction)
