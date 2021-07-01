
import sys
import numpy as np
import transformers
import babyai.utils as utils

try:
	import dash
	import dash_core_components as dcc
	import dash_html_components as html
	import plotly.express as px
	from dash.dependencies import Input, Output
except:
	print('To display the environment in a window, please install dash/plotly, eg:')
	print('pip3 install dash pandas statsmodels')
	sys.exit(-1)


class EnvRendererWrapper():
	def __init__(self, app, env, agent, manual_mode = False, probes_mode = False, update_freq = .1):
		# self.app = dash.Dash(__name__)
		self.app = app
		self.env = env
		self.obs = env.reset()
		self.agent = agent
		self.probes_mode = probes_mode
		self.manual_mode = manual_mode

		if not isinstance(self.agent.obss_preprocessor, utils.TransformerObssPreprocessor):
			self.lookup = {i:w for w,i in self.agent.obss_preprocessor.vocab.vocab.items()}

		self.step = 0
		self.episode_num = 0
		self.done = False

		graphs = [html.Div(children = [
			html.H3(id = 'mission-text', style={'margin-bottom': 20}),
			html.Div(children = [dcc.Graph(id = 'environment-vis', figure = self.draw_env())]),
			dcc.Interval(id = 'interval-generator', interval = update_freq * 1000)])]
		outcbs = [Output('environment-vis', 'figure'), Output('mission-text', 'children')]

		if manual_mode:
			raise NotImplementedError
		
		if probes_mode:
			self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
			graphs.append(html.Div(children = [dcc.Graph(id = 'attention-vis', figure = self.draw_blank())], style = {'display': 'flex', 'flex-direction': 'column', 'flex': 1}))
			graphs.append(html.Div(children = [dcc.Graph(id = 'actions-vis', figure = self.draw_blank())], style = {'display': 'flex', 'flex-direction': 'column', 'flex': 1}))

			outcbs.append(Output('attention-vis', 'figure'))
			outcbs.append(Output('actions-vis', 'figure'))


		self.app.layout = html.Div(children= [
			html.H1(children = 'BabyAI demo'),
			html.Div(children = graphs, style = {'display': 'flex'})])
		self.app.callback(*outcbs, Input('interval-generator', 'n_intervals'))(self.run_step)


	def run_step(self, _):
		if self.manual_mode:
			updatefigs = None
		else:
			if self.done:
				print("Reward:", self.reward)
				updatefigs = [self.draw_env(), 'Mission: {}'.format(self.obs["mission"])]
				if self.probes_mode:
					updatefigs.append(self.draw_blank())
					updatefigs.append(self.draw_blank())
				self.episode_num += 1
				# env.seed(args.seed + self.episode_num)
				self.obs = self.env.reset()
				self.agent.on_reset()
				self.step = 0
				self.done = False
			else:
				self.step += 1

				updatefigs = [self.draw_env(), 'Mission: {}'.format(self.obs["mission"])]
				result = self.agent.act(self.obs, probes = self.probes_mode)
				self.obs, self.reward, self.done, _ = self.env.step(result['action'])

				if self.probes_mode:
					updatefigs.append(self.draw_probes(result['probes']))
					updatefigs.append(self.draw_actions(result['dist'].probs[0]))

				self.agent.analyze_feedback(self.reward, self.done)
				if 'dist' in result and 'value' in result:
					dist, value = result['dist'], result['value']
					dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
					print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
						self.step, self.obs["mission"], dist_str, float(dist.entropy()), float(value)))
				else:
					print("step: {}, mission: {}".format(step, self.obs['mission']))
			

		return updatefigs

	def draw_probes(self, probes):
		if type(probes['encoded_inputs']) == dict:
			instructions = self.tokenizer.decode(probes['encoded_inputs']['input_ids'][0]).split()
		else:
			instructions = [self.lookup[i] for i in probes['encoded_inputs'].cpu().numpy()[0]]
		att_weights = probes['attention'].cpu().numpy().reshape((len(instructions), 1))
		# print(att_weights)
		
		fig = px.imshow(att_weights, title = 'Attention wights', color_continuous_scale = 'plasma')
		fig.update_xaxes(showticklabels=False)
		fig.update_yaxes(tickvals = np.arange(len(instructions)), ticktext = instructions)
		return fig

	def draw_actions(self, actions):
		lacts = len(actions)
		actions = actions.cpu().numpy().reshape((lacts, 1))

		fig = px.imshow(actions, title = 'Actions', color_continuous_scale = 'plasma')
		fig.update_xaxes(showticklabels=False)
		fig.update_yaxes(tickvals = np.arange(lacts), ticktext = [str(self.env.Actions(i)) for i in range(lacts)])
		return fig

	def draw_env(self):
		img = self.env.render(mode = 'rgb_array')
		fig = px.imshow(img)
		if len(img) > 500:
			fig.layout.width = len(img)*1.25
			fig.layout.height = len(img)*1.25
		fig.update_xaxes(showticklabels=False)
		fig.update_yaxes(showticklabels=False)
		return fig

	def draw_blank(self):
		img = px.imshow(np.zeros((1,1)))
		return img

	def render(self, probes = None, actions = None, **kwargs):
		if probes is not None:
			self.draw_probes(probes, actions)
		return self.env.render(**kwargs)

