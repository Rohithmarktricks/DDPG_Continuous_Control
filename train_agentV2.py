'''module train_agentV2.py contains the funcions to 
train the version2 of the training algorithm.
This trains the 20 agents that are linked to the single brain.

@author: Rohith Banka
'''

import argparse
import warnings
import numpy as np
from models import Actor, Critic
from time import strftime
from unityagents import UnityEnvironment
import torch
from collections import deque
from ddpg_agentV2 import Agent



def train_ddpg_agents(env, brain_name, agent, n_episodes=1000, max_t=10000,
                        solved_score=31.0, consec_episodes=100, print_every=1,
                        train_mode=True):
    '''Main function to train the 20 agents.

    Params:
    =======
    env             (UnityEnvironment Object): Reacher environment object
    brain_name      (key):                     Brain name to load the brain of Unity ML agent
    agent           (DDPG object):             DDPG (Actor Critic) agent
    n_episodes      (int):                     Total number of episodes to train the agent
    max_t           (int):                     Length of trajectory in the episode.
    required_score  (int):                     Score limit defined by Udacity to declare that the agent has solved the environment
    steps_for_avg   (int):                     To calculate the average of rewards over 100 episodes.
    print_every     (int):                     To print the status.
    train_mode      (boolean):                 True to indicate that the agent is in training mode.
    '''

    mean_scores=[]
    min_scores = []
    max_scores = []

    best_score = -np.inf
    scores_window = deque(maxlen=consec_episodes)
    moving_avgs = []

    print(f"Training 20 agents linked to single Brain, for {n_episodes} episodes and each episode(trajectory) has {max_t} steps !!!")
    print('===========================================================================================================================')

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # save the experience to replay buffer
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)

            states = next_states
            scores += rewards

            if np.any(dones):
                break

        min_scores.append(np.min(scores))
        max_scores.append(np.max(scores))
        mean_scores.append(np.mean(scores))
        scores_window.append(mean_scores[-1])
        moving_avgs.append(np.mean(scores_window))

        if i_episode % print_every == 0:
            print(f"Agents score - Episode: {i_episode} - \t Min: {round(min_scores[-1], 4)} - \t Max score: {round(max_scores[-1], 4)} - \t Mean score: {round(mean_scores[-1], 4)} - \t Moving Avg score: {round(moving_avgs[-1], 4)}")

        if moving_avgs[-1] >= solved_score and i_episode >= consec_episode:
            print(f"Reacher environment has been solved in {i_episode} episodes. \
                    Moving Average Score over last 100 episodes: {moving_avgs[-1]}")

            break

    start_time = strftime("%Y%m%d-%H%M%S")
    actor_network_path = f"saved_models/actor_20_ddpg_{start_time}.pth"
    critic_network_path = f"saved_models/critic_20_ddpg_{start_time}.pth"

    torch.save(agent.actor_local.state_dict(), actor_network_path)
    torch.save(agent.critic_local.state_dict(), critic_network_path)
    print("Saved actor_local and critic_local network weights in saved_models directory")

    min_scores_filename = f"scores/ddpg_20agents_min_score_{start_time}.csv"
    np.savetxt(min_scores_filename, min_scores, delimiter=',')
    max_scores_filename = f"scores/ddpg_20agents_max_score_{start_time}.csv"
    np.savetxt(max_scores_filename, max_scores, delimiter=",")
    mean_scores_filename = f"scores/ddpg_20agents_mean_score_{start_time}.csv"
    np.savetxt(mean_scores_filename, mean_scores, delimiter=",")
    mov_avgs_scores_filename = f"scores/ddpg_20agents_mvg_score_{start_time}.csv"
    np.savetxt(mov_avgs_scores_filename, moving_avgs, delimiter=",")
    print("Saved Min, Max, Mean, Moving Average(100 episodes) scores of 20 agents in scores folder")


    return mean_scores, moving_avgs



def main_20():
    """Main function to invoke the training of the DDPG agent"""
    parser = argparse.ArgumentParser(description="Train DDPG 20 agents linked to single Brain",
                                    usage="python train_agent_v2.py <Location of Reacher Env> <num_episodes> <max steps in every episode>")
    parser.add_argument("location", type=str, help="Location of the Reacher environment")
    parser.add_argument("num_episodes", type=int, help="Number of episodes to train the agent")
    parser.add_argument("max_steps", type=int, help="Number of steps in a single episode")

    # namespace
    namespace = parser.parse_args()

    # arguments
    location_20 = namespace.location
    num_episodes = namespace.num_episodes
    max_steps = namespace.max_steps
    
    env = UnityEnvironment(file_name=location_20)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents: ', num_agents)
    
    action_size = brain.vector_action_space_size
    print('Size of each action: ', action_size)
    
    states = env_info.vector_observations
    state_size = states.shape[-1]
    
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    
    print('The state for the first agent looks like:', states[0])

    
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)
    mean_scores, moving_avgs = train_ddpg_agents(env, brain_name, agent, n_episodes=num_episodes, max_t=max_steps)


if __name__ == '__main__':
    main_20()