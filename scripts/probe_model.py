
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
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from adjustText import adjust_text
import transformers

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
	embs = acmodel.to('cuda').word_embedding(torch.tensor(ipt, device = torch.device('cuda')))
	embs = np.squeeze(embs.cpu().detach().numpy())
	visualize_embs(embs, vocab)
	

def probe_vocab_from_model(model):
	vocab = utils.get_vocab_path(model)
	with open(vocab) as jin:
		vocab = json.load(jin)
	embs = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to('cuda').get_input_embeddings()
	ipt = list(vocab.values())
	embs = embs(torch.tensor(ipt, device = torch.device('cuda'))).cpu().detach().numpy()
	visualize_embs(embs, vocab)


def visualize_embs(embs, vocab):
	pca = PCA(n_components = 2)
	red = pca.fit_transform(embs)

	# tsne = TSNE(n_components = 2, perplexity = 30, learning_rate = 20, n_iter = 5000)
	# red = tsne.fit_transform(embs)

	fig,ax = plt.subplots(figsize = (12,12))
	X = red[:,0]
	Y = red[:,1]
	annotations = []
	for i, n in enumerate(vocab.keys()):
		ax.scatter(X[i], Y[i], s = 30, c = 'blue', alpha = 1)
		annotations.append(ax.text(X[i], Y[i], n, fontsize = 12, alpha = 1))
	adjust_text(annotations)

	fig.tight_layout()

	plt.show()


def probe_sents_transformer():
	colors = ['grey', 'green', 'purple', 'yellow', 'blue', 'red']
	combs = itertools.product(['a', 'the'], ['key', 'box', 'ball'])
	commands = [['pick', 'up'], ['go', 'to']]
	sents = [cmd + list(c) for c in combs for cmd in commands]

	sentlabels = [' '.join(sent[-3:]) for sent in sents]
	sents = [' '.join(s) for s in sents]

	model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').requires_grad_(False).to(torch.device('cuda'))
	tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

	fdict = tokenizer(sents, return_tensors = 'pt', padding = True).to(torch.device('cuda'))
	embs = model(**fdict).last_hidden_state
	embs = np.take(embs.cpu().detach().numpy(), 3, axis = 1)

	plot_embs(embs, sents)


def probe_sents_transformer_2objs():
	colors = ['grey', 'green', 'purple', 'yellow', 'blue', 'red']
	shapes = ['key', 'box', 'ball']
	combs = list(itertools.product(colors, shapes[:1], colors, shapes[1:]))
	combs += list(itertools.product(colors, shapes[1:], colors, shapes[:1]))

	sents = [['put', 'the'] + list(c[:2]) + ['next', 'to', 'the'] + list(c[2:]) for c in combs]

	sentlabels = [' '.join(c[:2]) + ' -> ' + ' '.join(c[2:]) for c in combs]
	cmap = {'box': 'green', 'ball': 'blue', 'key': 'red'}
	colorlabels = [cmap[c[1]] for c in combs]
	sents = [' '.join(s) for s in sents]

	model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').requires_grad_(False).to(torch.device('cuda'))
	tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

	fdict = tokenizer(sents, return_tensors = 'pt', padding = True).to(torch.device('cuda'))
	embs = model(**fdict).last_hidden_state
	embs = np.take(embs.cpu().detach().numpy(), 9, axis = 1)

	plot_embs(embs, sentlabels, colorlabels)


def probe_sents_babyai(model):
	colors = ['grey', 'green', 'purple', 'yellow', 'blue', 'red']
	combs = itertools.product(['a', 'the'], ['key', 'box', 'ball'])
	commands = [['pick', 'up'], ['go', 'to']]
	sents = [cmd + list(c) for c in combs for cmd in commands]

	vocab, acmodel = load_model(model)
	ipt = np.array([[vocab[w] for w in sent] for sent in sents])
	sentlabels = [' '.join(sent) for sent in sents]

	embs = acmodel.word_embedding(torch.tensor(ipt, device = torch.device('cuda')))
	embs = acmodel.instr_rnn(embs)[0]
	embs = np.take(embs.cpu().detach().numpy(), 2, axis = 1)

	plot_embs(embs, sentlabels)


def probe_conv(model):
	acmodel = utils.load_model(model)
	convweights = acmodel.image_conv.get_submodule('0').state_dict()['weight']
	convweights = torch.clip(convweights, -.1, .1).cpu().detach().numpy()

	fig, axs = plt.subplots(8, 16, sharex = True, sharey = True)
	fig.subplots_adjust(hspace = 0, wspace = 0)

	for i in range(8):
		for j in range(16):
			idx = i*16 + j
			img = convweights[idx]
			img = np.moveaxis(img, 0, -1)
			img = img * 5 + .5
			im = axs[i][j].imshow(img)

	plt.show()



def plot_embs(embs, sentlabels, colorlabels = None):
	if colorlabels is None:
		colorlabels = sentlabels
	colors = ['grey', 'green', 'purple', 'yellow', 'blue', 'red']
	shapes = {'key': '1', 'box': 's', 'ball': 'o'}
	pca = PCA(n_components = 2)
	red = pca.fit_transform(embs)

	fig,ax = plt.subplots(figsize = (12,12))
	ax.set_facecolor('black')
	X = red[:,0]
	Y = red[:,1]
	annotations = []
	for i, n in enumerate(sentlabels):
		clr = 'blue'
		mkr = 'o'
		for c in colors:
			if c in colorlabels[i]:
				clr = c
				break
		# for s, m in shapes.items():
		# 	if s in n:
		# 		mkr = m
		# 		break
		ax.scatter(X[i], Y[i], marker = mkr, s = 30, color = COLORS[clr]/255, alpha = 1)
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
	probe_sents_babyai('Embodiment-PickupGotoLocalShapeSplits-v0_ppo_pixels_endpool_res_attgru_mem_seed467699879_21-06-08-19-01-41_best') 
	# probe_vocab('BabyAI-BossLevel-v0_IL_pixels_endpool_res_attgru_seed616214275_21-07-09-11-36-41_best')
	# probe_sents_transformer()

	# probe_sents_transformer_2objs()