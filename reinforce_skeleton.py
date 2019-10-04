from grid_world import *
import numpy as np
import matplotlib.pyplot as plt
import math


class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions, temperature):
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature

        # here are the weights for the policy
        self.weights = np.zeros((num_states, num_actions))
        

    # TODO: fill this function in
    # it should take in an environment state
    # return the action that follows the policy's distribution
    def act(self, state):
        pdf = np.zeros([self.num_actions])
        if (state < 0 or state >= self.num_states):
            raise Exception("Invalid state") 
        for a in range(self.num_actions):
        # soft max policy parameterization
            pdf[a] = np.exp((self.weights[state][a])/self.temperature)/np.sum(np.exp(self.weights[state][:]/self.temperature))
        #print(np.sum(np.exp(self.weights[state][:]/self.temperature)))
        action = np.random.choice(self.num_actions,1,p=pdf)
        return action


    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and action
    # returns the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return): ### need to use action and state!
        grad = np.empty([self.num_states, self.num_actions])
        pdf = np.empty([self.num_states, self.num_actions]) # probability of getting action a given state s
        temp = np.empty([self.num_states, self.num_actions]) # RENAME THIS VARIABLE
        
        # policy parameterization, which is an (self.num_states, self.num_actions) numpy array
        for s in range(self.num_states):
            for a in range(self.num_actions):
                pdf[s][a] = np.exp(self.weights[s][a]/self.temperature)/np.sum(np.exp(self.weights[s][:]/self.temperature))
        
        #temp[state][action] = 1 ### for taking the gradient of the weights w.r.t to the weight parameter
        grad = self.weights - np.dot(self.weights,pdf)
        gradient = discounted_return*grad
        #print(np.shape(gradient))
        return gradient


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += step_size*grad
        


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):
    discount = np.zeros(len(rewards))
    temp = 0
    for t in reversed(range(0,len(rewards))):
        temp = temp*gamma+rewards[t]
        discount[t] = temp
    return discount.tolist()


# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate):
    episode_rewards = []
    
    # Remember to update your weights after each episode
    for e in range(num_episodes):
        state = env.reset()
        episode_log = []
        rewards = []
        score = 0
        
        done = False
        while True:
            # Sample from policy and take action in environment
            action = policy.act(state)
            next_state, reward, done = env.step(action)
            
            # Append results to the episode log
            episode_log.append([state, action, reward, next_state])
            state = next_state
            # Save reward in memory for self.weights updates
            rewards.append(reward)
            score += reward
            
            # If done, store results and calculate the gradients
            if done:
                episode_log = np.array(episode_log)
                rewards = episode_log[:,2].tolist()
                discount = get_discounted_returns(rewards, gamma)
                break
            
        # Calculate the gradients and perform policy weights update
        for i in range(len(np.vstack(episode_log[:,0]))):
            grads = policy.compute_gradient(episode_log[i,0], episode_log[i,1], discount[i])
            policy.gradient_step(grads, learning_rate) ### 
        #when should we be updating the weights? after each step or after each episode?
        
        # For logging the sum of the rewards for each episode
        episode_rewards.append(score)
    return episode_rewards


if __name__ == "__main__":
    gamma = 0.9
    num_episodes = 20000
    learning_rate = 1e-4
    env = GridWorld(MAP1)
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature=1)
    episode_rewards = reinforce(env, policy, gamma, num_episodes, learning_rate)
    #plt.plot(np.arange(num_episodes),episode_rewards)
    plt.plot(episode_rewards[0:100])
    plt.show()
    # rewards = np.zeros(num_episodes)
    # # gives a sample of what the final policy looks like
    # print("Rolling out final policy")
    # state = env.reset()
    # env.print()
    # done = False
    # while not done:
        # input("press enter to continue:")
        # action = policy.act(state)
        # state, reward, done = env.step(action) # do we need to store each reward in an array to plot later?
        # env.print()
        
    # Runs reinforce algorithm 20 times training on 20,000 episodes each time 
    # and counts the number of times the goal is reached
    # maxReward = np.amax(episode_rewards)
    # trials = 20
    # num_goals = 0
    # for t in range(trials):
        # episode_rewards = reinforce(env, policy, gamma, num_episodes, learning_rate)
        # for e in range(len(episode_rewards)):
            # if np.amax(episode_rewards) == maxReward:
                # num_goals+=1
    # print(num_goals)
