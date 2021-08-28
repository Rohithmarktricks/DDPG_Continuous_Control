''' Moudle utils.py
This module contains important functionalities related to device, plotting the scores of the agent.
@author: Rohith Banka.

'''


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from time import strftime


def get_device():
	'''This function helps to get the device for training the neural network.'''
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	return device

def get_plot(csv_filename):
	'''This function takes the CSV file (that contains the scores of the agent, untill the envioronment is solved) and plots the graph,
	saves it in the plots folder.'''

	df = pd.read_csv(csv_filename)

	# get x_ and y_ value for plotting.
	y_ = list(chain(*df.iloc[0:].values.tolist()))
	x_ = [i for i in range(len(y_))]

	scores_average_window = 100
	avg_scores = []
	start_time = strftime("%Y%m%d-%H%M%S")

	for i_episode in range(len(x_)):
		average_score = np.mean(y_[i_episode - min(i_episode, scores_average_window): i_episode+1])
		avg_scores.append(average_score)

	# find the episode where the agent has scores >= 30.0 over 100 consequtive episodes.
	avg_scores = np.array(avg_scores)
	amax = np.where(avg_scores >= 30)

	# plot the x_, y_, and mark the episode
	plt.plot(x_, y_, 'g', alpha=0.3)
	plt.plot(x_, avg_scores, 'g')
	amax = amax[0][0]

	x_lim, y_lim = plt.xlim(), plt.ylim()
	plt.plot([x_[amax], x_[amax], x_lim[0]], [x_lim[0], avg_scores[amax], avg_scores[amax]],
            linestyle='--')
	plt.plot(x_lim)
	plt.plot(y_lim)
	plt.xlabel('Episodes')
	plt.ylabel('Average Score over 100 episodes')

	# save the plot
	image_path = 'plots/ddpg_agent_avg_score_'+start_time+'img.jpg'
	plt.savefig(image_path)
	plt.show()
