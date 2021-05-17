
import numpy as np
import matplotlib.pyplot as plt
import transformers


class EnvRendererWrapper():
	def __init__(self, env, probes = False):
		self.env = env
		if probes:
			self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
			self.fig = plt.figure('Probes')

	def draw_probes(self, probes):
		instructions = self.tokenizer.decode(probes['encoded_inputs']['input_ids'][0]).split()
		att_weights = probes['attention'].cpu().numpy().reshape((len(instructions), 1))
		# print(att_weights)

		self.fig.clear()
		ax = self.fig.add_subplot()
		ax.imshow(att_weights, cmap = 'inferno')
		ax.set_yticks(np.arange(len(instructions)))
		ax.set_yticklabels(instructions)

	def render(self, probes = None, **kwargs):
		if probes is not None:
			self.draw_probes(probes)
		return self.env.render(**kwargs)

