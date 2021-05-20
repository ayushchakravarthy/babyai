
import numpy as np
import matplotlib.pyplot as plt
import transformers


class EnvRendererWrapper():
	def __init__(self, env, probes = False):
		self.env = env
		if probes:
			self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
			self.fig = plt.figure('Probes')

	def draw_probes(self, probes, actions):
		instructions = self.tokenizer.decode(probes['encoded_inputs']['input_ids'][0]).split()
		att_weights = probes['attention'].cpu().numpy().reshape((len(instructions), 1))
		# print(att_weights)
		lacts = len(actions)
		actions = actions.cpu().numpy().reshape((lacts, 1))

		self.fig.clear()
		ax1 = self.fig.add_subplot(1,2,1)
		act = ax1.imshow(actions, cmap = 'inferno')
		ax1.set_yticks(np.arange(lacts))
		ax1.set_yticklabels([str(self.env.Actions(i)) for i in range(lacts)])
		self.fig.colorbar(act, ax = ax1)

		ax2 = self.fig.add_subplot(1,2,2)
		att = ax2.imshow(att_weights, cmap = 'inferno')
		ax2.set_yticks(np.arange(len(instructions)))
		ax2.set_yticklabels(instructions)
		self.fig.colorbar(att, ax = ax2)

	def render(self, probes = None, actions = None, **kwargs):
		if probes is not None:
			self.draw_probes(probes, actions)
		return self.env.render(**kwargs)

