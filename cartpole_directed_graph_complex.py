#    This reinforcement learning implementation shows a way to solve the cartpole 
#    environment of OpenAI gym.
#    Copyright (C) 2018 Julius Zauleck
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version. This copyright notice may not be
#    removed.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <https://www.gnu.org/licenses/>.

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
         
for i in range(10000000):
    env.render() # omit the visualization to speed up simulation
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
    # If the last step resulted in failure, the number of concurrent successful steps is saved and the statistics of states leading to failure are updated, if the step before the last did not also result in failure.
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
