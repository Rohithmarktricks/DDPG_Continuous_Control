import torch
from time import strftime
import numpy as np
from collections import deque
from train_agent import get_environment_info, get_agent
import argparse


def train_twenty_agents(env, brain_name, agent, n_episodes=1000, max_t=10000,
                        solved_score=31.0, consec_episode=100, print_every=1,
                        train_mode=True):
	mean_scores = []
	min_scores = []
	max_scores = []

	best_score = -np.inf
	scores_window = deque(maxlen=consec_episode)
	moving_avgs = []

	for i_episode in range(1, n_episodes + 1):
		env_info = env.reset(tarin_mode=train_mode)[brain_name]
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
				agent.step(state, action, reward, next_state, done)

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
			print(f"Episode: {i_episode} - \t Min Score of Agent: {min_scores[-1]} - \t \
			Max Score of Agent: {max_scores[-1]} -/ \
				\t Mean score of Agent: {mean_scores[-1]} - \t Moving Avg score of Agent: {moving_avgs[-1]}")

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


def main():
	"""Main function to invoke the training of the DDPG agent"""

	parser = argparse.ArgumentParser(description="Train DDPG Agent",
	                                 usage="python train_agent.py <path to Reacher Environment>")
	parser.add_argument('location', type=str, help="Input location of the Reacher Environment")

	# get the namespace
	namespace = parser.parase_args()

	# get the args
	location = namespace.location

	# get the agent information
	env, brain_name, brain, action_size, state_size = get_environment_info(location)

	# Initialize the agent with state_size and action_size of the environment.
	agent = get_agent(state_size, action_size)

	# Train the agent.
	mean_scores, moving_avgs = train_twenty_agents(env, brain_name, agent)
