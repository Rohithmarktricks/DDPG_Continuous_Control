"""
This module contains all the required functions to train the agent.
@author: Rohith Banka
Initial code has been taken from module that I have developed for Navigation DQN project.
Reference: https://github.com/Rohithmarktricks/Navigation_DQN/blob/main/train_agent.py

Unity ML agents have Academy and brains.

Academy: This element in Unity ML, orchestrates agents and their decision-making process.
Brain: We train the agent by optimizing the policy called "Brain". We control this brain using Python API.

For further information, Please refer to the following Medium article:
https://towardsdatascience.com/an-introduction-to-unity-ml-agents-6238452fcf4c


"""

# import the required modules.
import torch
import numpy as np
from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment
from time import strftime
import argparse
import warnings


def get_environment_info(location):
    """To get the information about the environment from the given location of Unity ML agent"""
    env = UnityEnvironment(file_name=location)

    # We check for the first brain available, and set it as the default brain, we will be controlling using Python API.
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # action size is the total number of distinct actions.
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations

    # state size is the total number of dimensions of the each state in the environment. In our case, it's 37.
    state_size = states.shape[1]
    print(f"Successfully Loaded Reacher environment from {location}")
    print(f"Number of agents: {num_agents}")
    print(f"Size of each action: {action_size}")
    print(f"There are {states.shape[0]} agents. Each observes a state with length; {state_size}")
    print(f"The state for the first agent looks like: {states[0]}")
    return env, brain_name, brain, action_size, state_size


def get_agent(state_size, action_size):
    """Initializes and returns the new agent"""
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)
    return agent


def train_ddpg_agent(env, brain_name, agent, num_episodes=1000, max_t=10000):
    """

    Trains the agent.

        Params:
        =======
        env: Unity ML Environment
        brain_name: the name of the first brain available.
        num_episodes: Total number of episodes to train the agent.
        max_t: total number of steps in the trajectory to train the agent.

    """

    scores = []
    scores_deque = deque(maxlen=100)

    '''
    1. Reset the environment at the beginnign of each episode.
    2. Get the current state. i.e., s(t)
    3. Start the agent with above state as initial_state and explore the environment for max_t steps.
        a. Use the agent's act() method to get the actions.
        b. Pass it to the environment to take the step.
        c. Get the reward(r) and the next_state[s(t+1)] of the environment for action (a)
        d. Invoke agents step() method, that internally invokes learn() method.
        e. Update the total reward and set s(t) <- s(t+1)

    Repeat steps 1 to 3 until the episode is done.
    '''
    print(f"Training 1 agent linked to single Brain, for {num_episodes} episodes and each episode(trajectory) has {max_t} steps !!!")
    print('========================================================================================================================')
    for i_episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done, t)  # take step with agent (including learning)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        scores_deque.append(score)
        scores.append(score)

        print(f"Episode: {i_episode} \t Score: {score} \t Average Score: {np.mean(scores_deque)}")
        print(f"Episode: {i_episode} \t Average Score: {np.mean(scores)}")

        # Checks if the average of the reward over 100 consecutive episodes >= 31.0
        if np.mean(scores_deque) >= 31.0:
            print(f"Reacher Environment has been solved by training DDPG agent!!")
            print(f"Episodes taken {i_episode} \t Average Score: {np.mean(scores_deque)}")

    start_time = strftime("%Y%m%d-%H%M%S")
    print(f"saving Actor Critic models weights")
    actor_network_path = 'saved_models/checkpoint_actor_ddpg_' + start_time + '.pth'
    print(f"Saved Actor network weights at {actor_network_path}")
    critic_network_path = 'saved_models/checkpoint__ddpg_critic_' + start_time + '.pth'
    print(f"Saved Critic network weights at {critic_network_path}")

    torch.save(agent.actor_local.state_dict(), actor_network_path)
    torch.save(agent.critic_local.state_dict(), critic_network_path)

    # save the scores in the CSV file.
    scores_filename = "scores/ddpg_agent_score_" + start_time + ".csv"
    np.savetxt(scores_filename, scores, delimiter=",")
    print(f"Saved the scores of the agent in scores folder {scores_filename}")

    # close the environment.
    env.close()
    print("Closed the Reacher Environment")


def main():
    """Main function to invoke the training of the DDPG agent"""

    parser = argparse.ArgumentParser(description="Train DDPG Agent",
                                     usage="python train_agent.py <path to Reacher Environment>")
    parser.add_argument('location', type=str, help="Input location of the Reacher Environment")
    parser.add_argument('episodes', type=int, help="Number of episodes for training the agent")
    parser.add_argument('steps', type=int, help="Number of steps in a single episode")

    # get the namespace
    namespace = parser.parse_args()

    # get the args
    location = namespace.location
    episodes = namespace.episodes
    steps = namespace.steps

    warnings.filterwarnings("ignore")
    # get the agent information
    env, brain_name, brain, action_size, state_size = get_environment_info(location)

    # Initialize the agent with state_size and action_size of the environment.
    agent = get_agent(state_size, action_size)

    # Train the agent.
    train_ddpg_agent(env, brain_name, agent, num_episodes=episodes, max_t=steps)


if __name__ == '__main__':
    main()
