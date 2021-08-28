'''Moduel train_agent.py to train the DDPG agent
@author: Rohith Banka

Initial code has been taken from module that I have developed for Navigation DQN project.
Reference: https://github.com/Rohithmarktricks/Navigation_DQN/blob/main/train_agent.py
'''

import numpy as np
from time import strftime
import argparse
import warnings
from unityagents import UnityEnvironment
from ddpg_agent import Agent
from collections import deque

def get_environment_info(location):
    '''Loads the environment
    Prams:
    =====
    location (string): Location of the Reacher.exe environment.

    Output:
    ======
    environement info(Tuple): (env, env_info, brain_name, state_size, action_size)
    '''
    env = UnityEnvironment(file_name=location)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents: ', num_agents)

    action_size = brain.vector_action_space_size
    print('Size of each action: ', action_size)

    states = env_info.vector_observations
    state_size = states.shape[1]

    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like: ', states[0])

    return env, env_info, brain_name, brain, state_size, action_size


def get_agent(state_size, action_size, seed=10):
    '''Returns the DDPG Agent
    
    Params:
    ======
    state_size(int): Dimension of the state space
    action_size(int): Dimension of the action space
    seed(int): Number to preserver the configuation.

    '''
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)
    return agent


def train_ddpg(env, agent, brain_name, num_episodes=1000, max_t=10000):
    '''Trains the ddpg agent

    Params:
    ======
    env (UnityEnvironment object): Object that contains the Reacher Environment.
    agent (DDPG Agent object): The DDPG (Actor-Critic) PyTorch DNN
    brain_name (Unity ML brain name for the agent): This is responsible for the decision.
    num_episodes (int): total number of episodes that the agent has to be trained for.
    max_t (int): The total number of time steps in a single episode. This is the length of the trajectory

    '''
    scores = []
    scores_deque = deque(maxlen=100)
    avg_scores = []

    print(f'Training a DDPG agent for {num_episodes} episodes. Each episode has {max_t} steps!!')
    print('================================================================\n')

    for i_episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        avg_scores.append(np.mean(scores_deque))

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 31.0:
            print('\nEnvironment solved in {:d} Episodes \tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break

    start_time = strftime("%Y%m%d-%H%M%S")
    actor_network_path = f"actor_ddpg_{start_time}.pth"
    critic_network_path = f"critic_ddpg_{start_time}.pth"

    torch.save(agent.actor_local.state_dict(), actor_network_path)
    torch.save(agent.critic_local.state_dict(), critic_network_path)
    print("Saved actor_local and critic_local network weights in saved_models directory")

    scores_filename = f"ddpg_agent_scores_{start_time}.csv"
    np.savetxt(scores_filename, scores, delimiter=",")
    avg_scores_filename = f"ddpg_agent_avg_score_{start_time}.csv"
    np.savetxt(avg_scores_filename, avg_scores, delimiter=",")
    print("Saved the scores of the ddpg agent in scores folder")



def main():
    '''Main trigger method'''
    parser = argparse.ArgumentParser(description="Train DDPG agent",
                                    usage="python train_agent.py <path to Reacher Env> <episodes> <steps>")
    parser.add_argument("location", type=str, help="Input location of the Reacher Environment")
    parser.add_argument("episodes", type=int, help="Number of episodes to train the agent")
    parser.add_argument("steps", type=int, help="Number of steps in a single episode")

    # get the namespace
    namespace = parser.parse_args()

    # get the arguments
    location = namespace.location
    episodes = namespace.episodes
    steps = namespace.steps

    warnings.filterwarnings("ignore")

    env, env_info, brain_name, brain, state_size, action_size = get_environment_info(location)

    agent = get_agent(state_size, action_size, seed=20)
    train_ddpg(env, agent, brain_name, num_episodes=episodes, max_t=10000)

if __name__ == '__main__':
    main()