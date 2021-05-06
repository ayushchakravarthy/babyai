
import os
import csv
import json
import time
import datetime
import itertools

import torch
import numpy as np
import gym
import subprocess
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from adjustText import adjust_text

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from gym_minigrid.minigrid import COLORS


def load_model(model):
	acmodel = utils.load_model(model)
	vocab = utils.get_vocab_path(model)
	with open(vocab) as jin:
		vocab = json.load(jin)
	return vocab, acmodel


def probe_vocab(model):
	vocab, acmodel = load_model(model)
	ipt = list(vocab.values())
	embs = acmodel.word_embedding(torch.tensor(ipt, device = torch.device('cuda')))
	embs = np.squeeze(embs.cpu().detach().numpy())

	pca = PCA(n_components = 2)
	red = pca.fit_transform(embs)

	fig,ax = plt.subplots(figsize = (12,12))
	X = red[:,0]
	Y = red[:,1]
	annotations = []
	for i, n in enumerate(vocab.keys()):
		ax.scatter(X[i], Y[i], s = 30, c = 'blue', alpha = 1)
		annotations.append(ax.text(X[i], Y[i], n, fontsize = 12, alpha = 1))

	fig.tight_layout()

	plt.show()


def probe_sents(model):
	colors = ['grey', 'green', 'purple', 'yellow', 'blue', 'red']
	combs = itertools.product(['a', 'the'], colors, ['key', 'box'])
	sents = [['go', 'to'] + list(c) for c in combs]

	vocab, acmodel = load_model(model)
	ipt = np.array([[vocab[w] for w in sent] for sent in sents])
	sentlabels = [' '.join(sent[-3:]) for sent in sents]

	embs = acmodel.word_embedding(torch.tensor(ipt, device = torch.device('cuda')))
	embs = acmodel.instr_rnn(embs)[0]
	embs = np.take(embs.cpu().detach().numpy(), 3, axis = 1)

	pca = PCA(n_components = 2)
	red = pca.fit_transform(embs)

	fig,ax = plt.subplots(figsize = (12,12))
	ax.set_facecolor('black')
	X = red[:,0]
	Y = red[:,1]
	annotations = []
	for i, n in enumerate(sentlabels):
		clr = 'black'
		for c in colors:
			if c in n:
				clr = c
				break
		ax.scatter(X[i], Y[i], s = 30, color = COLORS[clr]/255, alpha = 1)
		annotations.append(ax.text(X[i], Y[i], n, color = COLORS[clr]/255, fontsize = 12, alpha = 1))

	adjust_text(annotations)

	fig.tight_layout()

	plt.show()


def sents_color_linreg(model):
	vocab, acmodel = load_model(model)

	combs = itertools.product(['a', 'the'], ['key', 'box'])
	for cmb in combs:
		sents = [['go', 'to', cmb[0], c, cmb[1]] for c in COLORS]
		ipt = np.array([[vocab[w] for w in sent] for sent in sents])

		embs = acmodel.word_embedding(torch.tensor(ipt, device = torch.device('cuda')))
		embs = acmodel.instr_rnn(embs)[0]
		embs = np.take(embs.cpu().detach().numpy(), 4, axis = 1)

		X = np.array(list(COLORS.values()))
		reg = LinearRegression().fit(X, embs)
		print(' '.join(cmb))
		print('Color linreg score: {}'.format(reg.score(X, embs)))


if __name__ == '__main__':
	# probe_sents('BabyAI-GoToLocal-v0_ppo_bow_endpool_res_gru_mem_seed1_21-04-25-22-57-07_pretrained_BabyAI-GoToLocal-v0_ppo_bow_endpool_res_gru_mem_seed4949_21-04-26-10-07-17_best')
	probe_sents('BabyAI-GoToLocal-v0_ppo_pixels_endpool_res_gru_mem_seed1_21-04-26-22-11-26_best')