#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""

import argparse
import gym
import time

import babyai.utils as utils
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from renderer import EnvRendererWrapper

try:
    import dash
except:
    print('To display the environment in a window, please install dash/plotly, eg:')
    print('pip3 install dash pandas statsmodels')
    sys.exit(-1)


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")
parser.add_argument("--probes", action="store_true", default=False,
                    help="enable probes of instr attention")

args = parser.parse_args()

action_map = {
    "left"      : "left",
    "right"     : "right",
    "up"        : "forward",
    "p"         : "pickup",
    "pageup"    : "pickup",
    "d"         : "drop",
    "pagedown"  : "drop",
    " "         : "toggle"
}

assert args.model is not None or args.demos is not None, "--model or --demos must be specified."
# if args.seed is None:
#     args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment
env = gym.make(args.env)
if args.model is not None and 'pixel' in args.model:
    env = RGBImgPartialObsWrapper(env)
env.seed(args.seed)

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# Define agent
agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)

# Run the agent

done = True

action = None

def keyDownCb(event):
    global obs

    keyName = event.key
    print(keyName)

    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in action_map and keyName != "enter":
        return

    agent_action = agent.act(obs)['action']

    # Map the key to an action
    if keyName in action_map:
        action = env.actions[action_map[keyName]]

    # Enter executes the agent's action
    elif keyName == "enter":
        action = agent_action

    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)
    if done:
        print("Reward:", reward)
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))


if args.manual_mode:
    env.window.reg_key_handler(keyDownCb)

if __name__ == '__main__':
    app = dash.Dash(__name__)
    renderer = EnvRendererWrapper(app, env, agent, args.manual_mode, args.probes, args.pause)
    app.run_server(debug=True)