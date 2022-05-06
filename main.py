#!/home/airlab/miniconda3/envs/prob/bin/python

import argparse
import os
import random
import time

import numpy as np
from pandas import Categorical
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb

from ppo_agent import PPOAgent
from env import Env


def strtobool(string):
    if string in ["T","t","True","true"]:
        return True
    elif string in ["F","f","False","false"]:
        return False
    else:
        raise Exception(f"String {string} could not be safely converted to bool.")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="TurtleRLEnv-v0",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=None,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25_000,
        help='total timesteps of all experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--train', action="store_true", default=False,
        help="whether to train, otherwise play example with current model")
    parser.add_argument('--gui', action="store_true", default=False,
        help="whether to display gui, otherwise run in background")
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--checkpoint', type=str,
        help="the checkpoint of the model to load for testing")
    parser.add_argument('--use-max', action="store_true", default=False,
        help="whether to do exploration when testing, or always use the value maximizing policy, default is False: use exploration not max")

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    args = parser.parse_args()
    
    return args


def main(args):
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    if type(args.seed) == type(None):
        args.seed = np.random.randint(0,10e6)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = Env()

    # split the rest of initialization base on whether training or testing
    if args.train == True:
        # finish computing arguments and validate
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        remainder = args.batch_size % args.minibatch_size
        if remainder == 1:
            raise Exception("Singleton minibatches not allowed. Choose a different number.")
        if remainder != 0:
            raise Warning("Stray minibatches. Consider choosing a number that goes in evenly.")
        remainder = args.total_timesteps % args.batch_size
        if remainder != 0:
            raise Exception("Number of batches (steps*envs) must go in evenly to total timesteps.")

        # weights and biases
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        # agent setup
        args.model_dir = os.sep.join(["model",run_name])
        os.makedirs(args.model_dir)

        agent = PPOAgent(args, envs, device)
        agent.load_actor(args.checkpoint)

        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        print("main.py: agent.train called")
        agent.train(args.num_steps, optimizer, run_name=run_name)
    else:
        if type(args.checkpoint) == type(None) or os.path.exists(args.checkpoint) == False:
            raise Exception("You must supply a model via the checkpoint argument if you are not training.")

        agent = PPOAgent(args, envs, device)
        agent.load_model(args.checkpoint)
        
        ob = torch.tensor(envs.reset()).to(device)
        while True:

            if args.use_max:
                action = max(agent.actor(ob))
            else:
                action_dist = Categorical(logits=agent.actor(ob))
                action = action_dist.sample()

            ob, _, done, _ = envs.step(action)
            ob = torch.tensor(ob).to(device)

            if done:
                ob = torch.tensor(envs.reset()).to(device)
                time.sleep(1/30)
        

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
    
