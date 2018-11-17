# Reinforcement Learning in the Cartpole Environment of OpenAI's Gym
The scripts shown here deal with the CartPole environment of OpenAI's Gym. They implement autonomous agents that are able to learn how to solve it, i.e. balancing the pole within a stable configuration for at least 195 steps. The first implementation uses a neural network (NN) implementation of Q-learning. The other two explore an original reinforcement learning algorithm using a directed graph to represent the environment in two different stages of complexity.

## 1) Neural Q-learning
Q-learning is a common reinforcement learning technique, where the decision policy is updated using the rewards assigned to the outcomes of an action (if a step leads to failure or not) and estimates of the best option of the decision policy itself for the next step (a nice explanation can be found at https://en.wikipedia.org/wiki/Q-learning). Here, we will train the policies using a NN, where the states are used as input and for each possible action (two in this case) an output will be produced that can be interpreted as the quality of the resulting state performing the respective action. 

We start with the imports. We will be using keras to set up, train and use our NN model for predictions.
```
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
```

Next, we initialize a number of variables, most notably the discount factor needed for Q-learning, the percentage of random actions at simulation start (_experimentstart_) and the reduction of randomness during the simulation (_experimentreduction_).
```
episodes = 2000

discount = 0.95 
experimentstart = 1.0 
experimentreduction = 0.003 
batchsize = 32
punishment = -10
```

Now the environment is loaded and the input and output dimensions for the NN are determined.
```
env = gym.make('CartPole-v1')
statedim = env.observation_space.shape[0]
actiondim = env.action_space.n
```

We set up a NN of two densely connected hidden layers with 30 neurons each and optimize it using Adam.
```
nn = Sequential()
nn.add(Dense(30, input_dim=statedim, activation='relu'))
nn.add(Dense(30, activation='relu'))
nn.add(Dense(actiondim, activation='linear'))
nn.compile(loss='mse', optimizer=Adam(lr=0.001))
```

While this is not necessary, one can try and keep some failures in the training data, in order to never lose touch with potentially fatal moves. In our case we assure to keep at least 10 instances in our data. We also keep track of the average number of successful steps over the last 100 runs, to determine when the environment can be considered solved (surviving 195 consecutive steps).
```
minfails = 10  
history = deque(maxlen=3000)
failhistory = deque(maxlen=minfails)
lastrewards = deque(maxlen=100)

done = False
solved = False
```

There are two steps for each episode. In the first step the environment is run and the NN model is used to decide on the actions. Simultaneously, data of the run is stored in _history_. In the second part the model is trained on the collected data.
```
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, statedim])
```

At each step of an episode we first decide whether to choose a random action or use the prescribed one, we then perform the environment step and append the results to the training data. If the step results in failure the statistics of the run are gathered and the next can be started.
```
    for tick in range(500):
        # env.render()
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
```

After each simulation run, if the training data set is large enough, the NN is trained. For this the target value is determined by the reward and punishment and the prediction of the model for the next step according to the Q-learning technique. Finally, for the next run the randomness is decreased.
```
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
```

## 2) Simple Directed Graph Approach
An alternative approach is to divide the overall state space in segments that each represent a volume in the state space and create an algorithm that memorizes the outcomes for all the segments in combination with an action. These segments, which we will refer to as states for simplicity, can for this environment lead to either other states or to failure after an action is performed. As a result one may consider an algorithm that assigns values to the states based on their best outcomes and decides to perform actions that lead to the most valuable states. For the sake of simplicity the state space is segmented by using a discrete grid along each state observable in the following implementations.

As a first step we will only track states that lead to failure and if both actions from state A lead to states that resulted in failure earlier, state A will also be marked as resulting in failure. Here, we initialize the state grid in way that each state is assumed not to result in failure and once both actions from that state resulted in failure it is considered to fail. This approach will clearly lead to problems, but it will serve as an instructive starting point for a more complex approach.

We start with the imports again.
```
import gym
import math
import numpy as np
```

Now we define a function that generates a 1D grid as a numpy array.
```
def createGrid(gridmin, gridmax, Npoints):
    d = (gridmax - gridmin) / (Npoints - 1)
    grid = np.empty(Npoints)
    for i in range(0, Npoints):
        grid[i] = gridmin + i * d
    return grid
```

We define a function that gives us the index of the grid point being closest to a certain value. This will be used to identify the discrete state from the state observations returned by the `step(...)` function.
```
def matrixIndexFromGrid(val, grid):
    if (val < grid[0] or val > grid[-1]):
        return -1
    for i in range(0, grid.size):
        if (val - grid[i] < 0):
            return i - 1
```

Here we define the parameters of the state grids. These include the number of grid points along each state observable (position, velocity, angle, angular velocity), scaling parameters, min and max value. After that we set up each grid.
```
Ngrid = 8

gammascale = 4
gammadotscale = 4
xscale = 1
xdotscale = 2

gammamin = -math.pi / 180.0 * 15.0 / gammascale
gammamax = math.pi / 180.0 * 15.0 / gammascale
gammadotmin = -3.0 / gammadotscale
gammadotmax = 3.0 / gammadotscale
xmin = -2.4 / xscale
xmax = 2.4 / xscale
xdotmin = -1.8 / xdotscale
xdotmax = 1.8 / xdotscale

gammagrid = createGrid(gammamin, gammamax, Ngrid + 1)
gammadotgrid = createGrid(gammadotmin, gammadotmax, Ngrid + 1)
xgrid = createGrid(xmin, xmax, Ngrid + 1)
xdotgrid = createGrid(xdotmin, xdotmax, Ngrid + 1)
```

Now we initialize a 4th order tensor that has a value for each discrete state. This value is initialized as 0 and corresponds to the action that will be performed when the environment reaches that state. As a result, initially the same action is performed in all states. After a state lead to failure or to a state that previously resulted in failure, its value is updated from 0 to 1 or from 1 to -1. Here 1 indicates the second action and -1 indicates failure.
```
actionarray = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid]) 
actionarray = actionarray.astype(int)
```

Next we initialize a bunch of variables that we will need for the simulation loop. Most notably, we need to keep track if we want to use the last step to update _actionarray_, based on whether the simulation was just reset and we need an index array _index_ to access _actionarray_.
```
lastindex = np.zeros(4)
uselast = False
lasti = 0
lastrewards = np.zeros([100])
episodec = 0
solved = False
env = gym.make("CartPole-v1")
observation = env.reset()
done = False
index = [matrixIndexFromGrid(observation[0], xgrid), matrixIndexFromGrid(observation[1], xdotgrid),
         matrixIndexFromGrid(observation[2], gammagrid), matrixIndexFromGrid(observation[3], gammadotgrid)]
```
         
Now we start the simulation. In each step we pick the next action according to the current state and if the current run lead to failure we update the action corresponding to the previous state from 0 to 1 or from 1 to -1. We also track whether the environment is solved (surviving 195 consecutive steps) and reset if an action lead to failure.
```
for i in range(10000000):
    # env.render()

    action = actionarray[index[0], index[1], index[2], index[3]]
    # if action is -1 or the observations are outside the grid, reset
    if done or any(np.equal(index, -1)) or action == -1:
        episodec += 1
        lastrewards[0] = i - lasti
        lastrewards = np.roll(lastrewards, 1, axis=0)
        # print(sum(lastrewards)/len(lastrewards))
        if sum(lastrewards) / len(lastrewards) > 195.0 and not solved:
            print("solved after " + str(episodec) + " episodes")
            solved = True
        lasti = i
        observation = env.reset()
        done = False
        if uselast:
            if actionarray[lastindex[0], lastindex[1], lastindex[2], lastindex[3]] == 0:
                actionarray[lastindex[0], lastindex[1], lastindex[2], lastindex[3]] = 1 # if it failed and was 0, try 1 next time
            else:
                actionarray[lastindex[0], lastindex[1], lastindex[2], lastindex[3]] = -1 # if both failed, set to -1
        uselast = False
    else:
        observation, reward, done, info = env.step(action)
        uselast = True
    lastindex = index
    index = [matrixIndexFromGrid(observation[0], xgrid), matrixIndexFromGrid(observation[1], xdotgrid),
             matrixIndexFromGrid(observation[2], gammagrid), matrixIndexFromGrid(observation[3], gammadotgrid)]
```

We see that this approach usually solves this environment. However, if we let it run through we also see that in the end the environment is reset every iteration. This is mainly caused by two influences. First, the discrete grid can not capture the full state space precisely. Secondly, there is some randomness in the initial conditions. In combination this means that the actual state in the continuous state space after each step can have slight variations and still be identified as the same state on the discrete grid. As a result it can at one time lead to failure and at another time not. However, with this approach we exclude a state as soon as it fails once and after letting the environment run for a while every state leads to failure sooner or later so it will be excluded sooner or later until all states are excluded.

## 3) Complex Directed Graph Approach
Now let's go one step further. Given the finite number of states that we have due to our discrete grid, we can imagine the state space as a directed graph. Ideally in this graph, from each node that corresponds to a state 2 edges that correspond to the actions would leave and each lead to a node that also corresponds to a state. However, in our system an action from a start state can have several different resulting states. We can incorporate this in a simple statistic that approximates the statistically expected outcome of a certain action given a start state. As a result, one action from a state will be represented as a number of edges leading to the possible resulting nodes with each edge having a probability associated with it. We can then assign values to the nodes (which correspond to states) that are updated as the simulation is run based on the best expected outcome of the two possible actions. If we assign a low value to the node corresponding to failure, the value of the nodes leading faster (with fewer steps) to failure will decrease faster and a good strategy can be obtained by choosing the actions leading to nodes of higher value.

The first part of the initialization stays the same, as well as our two grid functions.
```
import gym
import math
import numpy as np
import random


def createGrid(xmin, xmax, Nx):
    dx = (xmax - xmin) / (Nx - 1)
    grid = np.empty(Nx)
    for i in range(0, Nx):
        grid[i] = xmin + i * dx
    return grid


def matrixIndexFromGrid(val, grid):
    if (val < grid[0] or val > grid[-1]):
        return -1
    for i in range(0, grid.size):
        if (val - grid[i] < 0):
            return i - 1


Ngrid = 8

gammascale = 4
gammadotscale = 4
xscale = 1
xdotscale = 2

gammamin = -math.pi / 180.0 * 15.0 / gammascale
gammamax = math.pi / 180.0 * 15.0 / gammascale
gammadotmin = -3.0 / gammadotscale
gammadotmax = 3.0 / gammadotscale
xmin = -2.4 / xscale
xmax = 2.4 / xscale
xdotmin = -1.8 / xdotscale
xdotmax = 1.8 / xdotscale

gammagrid = createGrid(gammamin, gammamax, Ngrid + 1)
gammadotgrid = createGrid(gammadotmin, gammadotmax, Ngrid + 1)
xgrid = createGrid(xmin, xmax, Ngrid + 1)
xdotgrid = createGrid(xdotmin, xdotmax, Ngrid + 1)
```

Next, we need one array to store all the values of the different states (_statevaluearray_), 2 arrays to store the edges that connect a start state with with a result state given an action (_actionarraygraph0_, _actionarraygraph1_) and 2 arrays that count the number of times the results occurred (_actionarraystats0_ and _actionarraystats1_). The edge array and its tracking array need a dimension for the number of result states that are tracked (_nconnect_) as well as a dimension for a 4-tuple corresponding to the indices of the result state on the grid. The number of times of an action from a state directly resulting in failure is counted in the separate arrays _failstats0_ and _failstats1_.
```
nconnect = 10
statevaluearray = np.ones([Ngrid, Ngrid, Ngrid, Ngrid]) # All the state values are initialized as 1.

actionarraygraph0 = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid, nconnect, 4]) # Action 0
actionarraygraph0 = actionarraygraph0.astype(int)
actionarraystats0 = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid, nconnect]) 
actionarraystats0 = actionarraystats0.astype(int)
failstats0 = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid]) 

actionarraygraph1 = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid, nconnect, 4]) # Action 1
actionarraygraph1 = actionarraygraph1.astype(int)
actionarraystats1 = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid, nconnect])
actionarraystats1 = actionarraystats1.astype(int)
failstats1 = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid])
```

Before going into the main loop we set a threshold the state values must have in order to not explore a new action, a bonus for every degree the state is removed from failure and a value associated with failure. In addition the same main loop variables as for the previous implementation are initialized.
```
threshold = 1 # Always explore unexplored actions
bonus = 0.00 
failvalue = -5 

lastindex = np.zeros(4)
uselast = False
lastaction = 0
lasti = 0
lastrewards = np.zeros([100])
episodec = 0
solved = False
env = gym.make("CartPole-v1")
observation = env.reset()
done = False
index = [matrixIndexFromGrid(observation[0], xgrid), matrixIndexFromGrid(observation[1], xdotgrid),
         matrixIndexFromGrid(observation[2], gammagrid), matrixIndexFromGrid(observation[3], gammadotgrid)]
```
         
Compared to the previous implementation the main loop is getting a bit crowded. While the same two things have to be done at every step, i.e. choosing the best action and updating the strategy, both are much more intricate.
```
for i in range(10000000):
    env.render()
```

First, we deal with the action decision. This is only relevant if the last step did not result in failure, otherwise the environment is reset. Here we differentiate between 4 cases. In the first the current state is new and there are no statistics which action leads to some other state. In this case a random action is chosen. In the second and third case there are statistics of resulting state for one of the two actions. For those the expected value of the next state is calculated and compared to _threshold_ to decide whether the result is good enough or if the other action should be tried. In the fourth case there are statistics for both actions and the better one is chosen. Note that here is a lot of room for improvement to explore the state space more efficiently.
```
    if not any(np.equal(index, -1)):
        # if both actions have not been tried
        if sum(actionarraystats0[index[0], index[1], index[2], index[3], :]) == 0 and failstats0[
                index[0], index[1], index[2], index[3]] == 0 \
                and sum(actionarraystats1[index[0], index[1], index[2], index[3], :]) == 0 and failstats1[
                index[0], index[1], index[2], index[3]] == 0:
            action = random.randint(0, 1)
        # if 0 action has not been tried
        elif sum(actionarraystats0[index[0], index[1], index[2], index[3], :]) == 0 and failstats0[
                index[0], index[1], index[2], index[3]] == 0:
            score1 = failvalue * failstats1[index[0], index[1], index[2], index[3]]
            norm = failstats1[index[0], index[1], index[2], index[3]]
            for connection in range(0, nconnect):
                tmp = actionarraygraph1[index[0], index[1], index[2], index[3], connection, :]
                score1 += statevaluearray[tmp[0], tmp[1], tmp[2], tmp[3]] * actionarraystats1[
                    index[0], index[1], index[2], index[3], connection]
                norm += actionarraystats1[index[0], index[1], index[2], index[3], connection]
            score1 = score1 / norm
            if score1 > threshold:
                action = 1
            else:
                action = 0
        # if 1 action has not been tried
        elif sum(actionarraystats1[index[0], index[1], index[2], index[3], :]) == 0 and failstats1[
                index[0], index[1], index[2], index[3]] == 0:
            score0 = failvalue * failstats0[index[0], index[1], index[2], index[3]]
            norm = failstats0[index[0], index[1], index[2], index[3]]
            for connection in range(0, nconnect):
                tmp = actionarraygraph0[index[0], index[1], index[2], index[3], connection, :]
                score0 += statevaluearray[tmp[0], tmp[1], tmp[2], tmp[3]] * actionarraystats0[
                    index[0], index[1], index[2], index[3], connection]
                norm += actionarraystats0[index[0], index[1], index[2], index[3], connection]
            score0 = score0 / norm
            if score0 > threshold:
                action = 0
            else:
                action = 1
        # if both options have been tried
        else:
            score1 = failvalue * failstats1[index[0], index[1], index[2], index[3]]
            norm = failstats1[index[0], index[1], index[2], index[3]]
            for connection in range(0, nconnect):
                tmp = actionarraygraph1[index[0], index[1], index[2], index[3], connection, :]
                score1 += statevaluearray[tmp[0], tmp[1], tmp[2], tmp[3]] * actionarraystats1[
                    index[0], index[1], index[2], index[3], connection]
                norm += actionarraystats1[index[0], index[1], index[2], index[3], connection]
            score1 = score1 / norm

            score0 = failvalue * failstats0[index[0], index[1], index[2], index[3]]
            norm = failstats0[index[0], index[1], index[2], index[3]]
            for connection in range(0, nconnect):
                tmp = actionarraygraph0[index[0], index[1], index[2], index[3], connection, :]
                score0 += statevaluearray[tmp[0], tmp[1], tmp[2], tmp[3]] * actionarraystats0[
                    index[0], index[1], index[2], index[3], connection]
                norm += actionarraystats0[index[0], index[1], index[2], index[3], connection]
            score0 = score0 / norm

            # decide on the action with the better score and update the value of the current state
            if score0 > score1:
                action = 0
                if score0 < 1 - bonus:
                    statevaluearray[index[0], index[1], index[2], index[3]] = score0 + bonus
                else:
                    statevaluearray[index[0], index[1], index[2], index[3]] = 1
            elif score1 > score0:
                action = 1
                if score1 < 1 - bonus:
                    statevaluearray[index[0], index[1], index[2], index[3]] = score1 + bonus
                else:
                    statevaluearray[index[0], index[1], index[2], index[3]] = 1
            # if both are equal, do a random action
            else:
                action = random.randint(0, 1)
                if score1 < 1 - bonus:
                    statevaluearray[index[0], index[1], index[2], index[3]] = score1 + bonus
                else:
                    statevaluearray[index[0], index[1], index[2], index[3]] = 1
    # If the last step resulted in failure, the number of concurrent successful steps is 
    # saved and the statistics of states leading to failure are updated, if the step before 
    # the last did not also result in failure.
    if done or any(np.equal(index, -1)):
        episodec += 1
        lastrewards[0] = i - lasti
        lastrewards = np.roll(lastrewards, 1, axis=0)
        print(str(episodec)+ " " + str(i-lasti))
        if sum(lastrewards) / len(lastrewards) > 195.0 and not solved:
            print("solved after " + str(episodec) + " episodes")
            solved = True
        lasti = i

        observation = env.reset()
        done = False
        if uselast:
            if lastaction == 0:
                failstats0[lastindex[0], lastindex[1], lastindex[2], lastindex[3]] += 1
            else:
                failstats1[lastindex[0], lastindex[1], lastindex[2], lastindex[3]] += 1
        uselast = False
```

In the second part, the statistics of the state space are updated as long as the last step was not a reset. For each action, it is checked whether the last step lead to a state that was already tracked as a result. If this is the case, the number of times it resulted in this state is incremented by one. If it is not the case and there is room for at least one other tracked result state, the resulting state is added and its count initialized at 1.  Finally, a step forward is performed.
```
    else:
        if uselast:
            if lastaction == 0:
                for connection in range(0, nconnect):
                    tmp = actionarraygraph0[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection, :]
                    if np.array_equal(tmp, index):
                        actionarraystats0[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection] += 1
                        break
                    if actionarraystats0[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection] == 0:
                        actionarraygraph0[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection, :] = index
                        actionarraystats0[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection] = 1
                        break
            else:
                for connection in range(0, nconnect):
                    tmp = actionarraygraph1[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection, :]
                    if np.array_equal(tmp, index):
                        actionarraystats1[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection] += 1
                        break
                    if actionarraystats1[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection] == 0:
                        actionarraygraph1[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection, :] = index
                        actionarraystats1[lastindex[0], lastindex[1], lastindex[2], lastindex[3], connection] = 1
                        break

        observation, reward, done, info = env.step(action)
        lastaction = action
        uselast = True
    # remember the last state
    lastindex = index
    index = [matrixIndexFromGrid(observation[0], xgrid), matrixIndexFromGrid(observation[1], xdotgrid),
             matrixIndexFromGrid(observation[2], gammagrid), matrixIndexFromGrid(observation[3], gammadotgrid)]
```
