'''Module to test the agent.
@author: Rohith Banka
Initial code has been taken from module that I have developed for Navigation DQN project.
Reference: https://github.com/Rohithmarktricks/Navigation_DQN/blob/main/test_agent.py

'''

# import modules
import torch
from train_agent import get_environment_info, get_agent
from argparse import ArgumentParser


def test_agent_in_env(env, agent, brain_name):
	"""This method tests teh agent in the environment"""

	# reset and get the environment details
	env_info = env.reset(train_mode=False)[brain_name]

	# Get the initial state of the agent.
	state = env_info.vector_observations[0]
	score = 0

	while True:

		# Get the actions.
		action = agent.act(state)

		# To get the environement information after the action.
		env_info = env.step(action)[brain_name]

		# get the next_state.
		next_state = env_info.vector_observations[0]
		# get the reward for the action.
		reward = env_info.rewards[0]

		# Boolean value to know if the agent has reached finished the episode.
		done = env_info.local_done[0]

		state = next_state
		score += reward

		# Break if finished the episode.
		if done:
			break

	print(f"The agent has been to score {round(score, 3)}..")

	env.close()
	print('Done with testing the agent, so closing the environment...')


def main():
	parser = ArgumentParser(description='Testing the agent',
	                        usage='python test_agent.py <location of the Unity ML environment> <path to '
	                              'actor_network.pth file>')
	parser.add_argument('location', type=str, help='Location of the Unity ML environment')
	parser.add_argument('actor_path', type=str,
	                    help="Location of the actor network models weights file. Please check saved_models folder.")

	# get the namespace
	namespace = parser.parse_args()

	# get the arguments
	location = namespace.location
	actor_path = namespace.actor_path

	env, brain_name, brain, action_size, state_size = get_environment_info(location)
	agent = get_agent(state_size=state_size, action_size=action_size)

	try:
		agent.network.load_state_dict(torch.load(actor_path))
		print("Loaded the weights successfully")
	except Exception as e:
		raise Exception(f"{e}")


if __name__ == "__main__":
	main()
