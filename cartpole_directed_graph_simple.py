import gym
import math
import numpy as np

def createGrid(gridmin, gridmax, Npoints):
    d = (gridmax - gridmin) / (Npoints - 1)
    grid = np.empty(Npoints)
    for i in range(0, Npoints):
        grid[i] = gridmin + i * d
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

actionarray = np.zeros([Ngrid, Ngrid, Ngrid, Ngrid]) 
actionarray = actionarray.astype(int)

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
         
for i in range(10000000):
    # env.render() # omit the visualization to speed up simulation

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
