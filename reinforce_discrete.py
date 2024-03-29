#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:55:11 2019

@author: matt
"""

from mountain_car import *
import matplotlib.pyplot as plt
import numpy as np
import pdb

class CustomPolicy(object):
    def __init__(self, num_states, action_grid):
        self.num_states = num_states
        self.action_grid = action_grid
        self.num_actions = len(self.action_grid)
        # here are the weights for the policy - you may change this initialization
        self.weights = np.random.random((self.num_states, self.num_actions))

    def softmax(self):
        num=np.exp(self.weights)
        pi=np.atleast_2d(1/np.sum(num,axis=1)).T*num
        return pi
    # it should take in an environment state
    # return the action that follows the policy's distribution
    def act(self, state):
        pi=self.softmax()
        action_index = np.random.choice(self.num_actions,1,p=pi[state,:])
        return self.action_grid[action_index][0], action_index[0]

    # computes the gradient of the discounted return
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return):
        pi=self.softmax()
        
        grad_left=np.zeros(self.weights.shape)
        grad_left[state,action]=1
        
        grad_right=np.zeros(self.weights.shape)
        grad_right[state,:]=pi[state,:]
        
        grad_mat=(discounted_return)*(grad_left-grad_right)
        return grad_mat


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights = self.weights + grad*step_size


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):

    discount_rewards = np.zeros(len(rewards))
    temp = 0
    for t in reversed(range(0,len(rewards))):
        temp = temp*gamma+rewards[t]
        discount_rewards[t] = temp
    return discount_rewards


# TODO: fill this function in
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate, state_grid):
    episode_rewards = []
    e=0
    while e<=num_episodes:
        state = env.reset()
        state = discretize(state, state_grid)

        episode_log = []
        iter = 0
        done = False

        while done == False and iter < 1000:
            # get action

            action,action_index = policy.act(state)
            # get next step, reward and chek whether reached goal
            next_state, reward, done, blah= env.step(action)
            # store the state, action, reward and next state
            episode_log.append([state, action_index, reward, next_state])

            state = discretize(next_state,state_grid)
            iter += 1


        episode_log = np.asarray(episode_log)
        rewards = episode_log[:,2]
        
        episode_rewards.append(np.sum(rewards))
        e+=1
#        print(done,iter,e,np.round(np.sum(rewards)))
        if e>5000 or done:
#            print('gradient step')            
            discount_rewards = get_discounted_returns(rewards, gamma)
            for t in range(0,len(episode_log)):
                grads = policy.compute_gradient(episode_log[t,0], episode_log[t,1],discount_rewards[t])
                policy.gradient_step(grads, learning_rate)

    return episode_rewards



def discretize(state, grid):

    s = np.zeros([len(state),1])
    nX=len(grid[0])+1
    nV=len(grid[1])+1
    for l in range(0,len(grid)):
#        print(l)
        s[l] = np.digitize(state[l], grid[0])
#    print(state, s)
    ind=np.reshape(np.arange(nV*nX),[nX,nV])
    state_index=ind[int(s[0]),int(s[1])]
    return state_index

def create_grid(low, high, bins):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

    return grid





if __name__ == "__main__":
    gamma = 0.9
    num_episodes = 1000
    learning_rate = 0.0001
    env = Continuous_MountainCarEnv()

    state_high = env.high_state
    state_low = env.low_state

    action_high = [env.max_action]
    action_low = [env.min_action]

    state_grid = create_grid(state_low, state_high, bins=(3,3))
    action_grid = np.linspace(action_low,action_high,3)

    R=[]
    for t in range(5):
        policy = CustomPolicy(9, action_grid)
        r=reinforce(env, policy, gamma, num_episodes, learning_rate, state_grid )
        R.append(r)
    
    r_min=np.min(np.array(R),axis=0)
    r_max=np.max(np.array(R),axis=0)
    plt.figure()
    plt.plot(r_min)
    plt.plot(r_max)
    
    plt.figure()
    for i in range(5):
        plt.plot(R[i])
