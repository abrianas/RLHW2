from mountain_car import *
import numpy as np
import pdb

class CustomPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        # here are the weights for the policy - you may change this initialization
        self.weights = np.random.random((self.num_states, self.num_actions))



    # TODO: fill this function in
    # it should take in an environment state
    # return the action that follows the policy's distribution
    def act(self, state):

        sigma = 1.0
        action = np.random.normal(np.dot(self.weights.T,state),sigma)

        return action


    # TODO: fill this function in
    # computes the gradient of the discounted return
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return):

        sigma = 1.0
        grad = ((action - np.dot(self.weights.T,state))/sigma**2)*state

        grad = grad*discounted_return
        # print("grad",grad)
        return grad


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):

        self.weights = self.weights + np.reshape(grad,[-1,1])*step_size


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

    for e in range(num_episodes):

        state = env.reset()
        state_d = discretize(state, state_grid)

        episode_log = []
        iter = 0
        done = False

        while done == False and iter < 1000:
            # get action

            action = policy.act(feature_state)
            # get next step, reward and chek whether reached goal
            next_state, reward, done, blah= env.step(action)
            # store the state, action, reward and next state
            episode_log.append([feature_state, action, reward, next_state])

            feature_state = rbf(next_state, centers, rbf_sigma)
            iter += 1


        episode_log = np.asarray(episode_log)
        rewards = episode_log[:,2]
        episode_rewards.append(np.sum(rewards))
        discount_rewards = get_discounted_returns(rewards, gamma)

        print(done,iter,e,np.sum(rewards))



        for t in range(0,len(episode_log)):

            grads = policy.compute_gradient(episode_log[t,0], episode_log[t,1],discount_rewards[t])
            policy.gradient_step(grads, learning_rate)


    return episode_rewards

def discretize(state, grid):

    s = np.zeros([len(state),1])

    for l in range(0,len(grid)):
        print(l)
        s[l] = grid[l][np.digitize(state[l], grid[0])]
    print(state, s)
    return s






def create_grid(low, high, bins)):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]

    return grid





if __name__ == "__main__":
    gamma = 0.9
    num_episodes = 20000
    learning_rate = 0.0001
    env = Continuous_MountainCarEnv()

    state_high = env.high_state
    state_low = env.low_state

    action_high = env.max_action
    aciton_low = env.min_action

    state_grid = state_create_grid(state_low, state_high, bins=(10,10))
    action_grid = create_grid(action_low, action_high, bins= 10))


    policy = CustomPolicy(2, 1)
    reinforce(env, policy, gamma, num_episodes, learning_rate, state_grid )

    # gives a sample of what the final policy looks like
    print("Rolling out final policy")
    state = env.reset()
    # env.print()
    done = False
    while not done:
        input("press enter to continue:")
        action = policy.act(state)
        state, reward, done, _ = env.step([action])
        # env.print()