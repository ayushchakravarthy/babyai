#!/usr/bin/env python3

import argparse
import gym
import time

import babyai.utils as utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin REQUIRED)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")

args = parser.parse_args()

assert args.model is not None or args.demos_origin is not None, "--model or --demos-origin must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Define agent

agent = utils.load_agent(args, env)

# Run the agent

done = True

while True:
    time.sleep(args.pause)
    renderer = env.render("human")

    if done:
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if done:
        print("Reward:", reward)

    if renderer.window is None:
        break