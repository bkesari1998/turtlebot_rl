#!/usr/bin/env python3

import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from model_cnn import Model


class PPOAgent(nn.Module):
    def __init__(self, args, envs, device):
        super(PPOAgent,self).__init__()

        self.critic = Model(1, 1.0)
        self.actor = Model(4, .01)

        self.args = args
        self.envs = envs
        self.device = device
        self.to(device)

    def get_value(self, x):
        return self.critic(x.float())

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x.float())
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        tmp1 = action
        tmp2 = probs.log_prob(action)
        tmp3 = probs.entropy()
        tmp4 = self.critic(x.float())
        return tmp1, tmp2, tmp3, tmp4

    def rollout(self, num_steps, num_envs, next_obs, next_done):
        #policy rollout is its own loop
        step = 0
        while step < num_steps:
            self.global_step += 1 * num_envs
            self.obs[step] = torch.tensor(next_obs).to(self.device)
            self.dones[step] = torch.tensor(next_done).to(self.device)
            
            '''
            ALGO LOGIC: action logic
            Actual rollout of policies... during the policy rollouts we do not need 
            '''
            with torch.no_grad():
                action, logprob, _, value = self.get_action_and_value(torch.tensor(next_obs).to(self.device))
                self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob


            '''
            Stepping the environment
            TRY NOT TO MODIFY: execute the game and log data.
            '''
            next_obs, reward, done = self.envs.step(action.cpu().numpy(), step)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            # current_step = self.rewards[step][0].item()
        

            #print('as just number', self.rewards[step][0].item())
            #print('*'*30)
            #print('at each step',self.rewards[step])
            #print('*'*30)
            next_obs = torch.tensor(np.asarray(next_obs)).to(torch.float32).to(self.device) 
            next_done = torch.tensor(done).to(self.device)
            self.next_obs = next_obs
            self.next_done = next_done
            #print('*'*30)
            #print('whole tensor',self.rewards)
            #print('*'*30)
            #if done == True:
            #    break
            #else:
            step += 1

            #'''
            #This loop gives us our whole episodic return and prints it out... there will be 25_000 time steps/whatever we put in total-timesteps
            #'''
            #for item in info:
            #    if "episode" in item.keys():
            #        print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
            #        self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
            #        self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.global_step)
            #        break
        
        self.writer.add_scalar("charts/episodic_return", np.sum(self.rewards.cpu().numpy()), self.global_step)
        self.writer.add_scalar("charts/episodic_length", step, self.global_step)    


    def advantage(self, num_steps, gamma, gae=False, gae_lambda=None):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.get_value(self.next_obs).reshape(1, -1)

            '''
            GAE or General advantage estimation... hellish and PPO specific... need to comment heavily...
            '''
            if gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

                '''
            This is the more common way to do advantage calculation and is much simpler
            returns are not the same and that should also be noted
                '''
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = self.rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - self.values

        return advantages, returns


    def train(self, num_steps, optimizer, run_name=""):
        num_envs = self.envs.env_fns

        # TRY NOT TO MODIFY: start the game
        start_time = time.time()
        self.global_step = 0
        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(num_envs).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size

        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        for update in range(1, num_updates + 1):

            # ALGO Logic: Storage setup (moved inside loop to account for early breaking)
            self.obs = torch.zeros((num_steps, num_envs) + self.envs.single_observation_space).to(self.device)
            self.actions = torch.zeros((num_steps, num_envs) + self.envs.single_action_space.shape).to(self.device)
            self.logprobs = torch.zeros((num_steps, num_envs)).to(self.device)
            self.rewards = torch.zeros((num_steps, num_envs)).to(self.device)
            self.dones = torch.zeros((num_steps, num_envs)).to(self.device)
            self.values = torch.zeros((num_steps, num_envs)).to(self.device)

            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates #fraction is one at the beginning and linearly decreases to 0 after all updates
                lrnow = frac * self.args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow#update learning rate w pytorch api... and at the end this is all one iteration of the training loop

            # policy rollout
            self.envs.reset()
            self.rollout(num_steps, num_envs, self.next_obs, self.next_done)

            # GAE or alternate advantage calculation
            advantages, returns = self.advantage(num_steps, self.args.gamma, gae=self.args.gae, gae_lambda=self.args.gae_lambda)

            # flatten variables to see the batch size
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            '''
            for training we take each indices of the the batch amd shuffle them for any given epoch
            '''
            # Optimizaing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):#looping through entire batch one minibatch at a time.. each minibatch == 128 random batch indices
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    '''
                    Training Fully begins!!!
                    
                    First a forward pass on mini batch observations
                    '''
                    _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])#training officially starts... mini batch actions only
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()#log ratio if this new log probabilities to policy rollout

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()# helps with debugging (look in PPO paper)
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]# helps with debugging (look in PPO paper)


                    '''
                    PPO adv normalization
                    '''
                    mb_advantages = b_advantages[mb_inds]
                    #print(mb_advantages)
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    #print(mb_advantages)

                    # Policy loss/ clipped policy objective
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss/value loss clipping
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,#original paper method... had to look up...
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()# apparently much more normal way to use it
                    
                    '''
                    entropy loss: entropy is the measure of chaos in a system
                    &
                    overall loss: maximazing entropy while minimizing the policy loss lets the agent explore still
                    '''
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef


                    #backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)# due to global gradient clipping we need this.
                    optimizer.step()

                if self.args.target_kl is not None:
                    if approx_kl > self.args.target_kl:
                        break



            #Early Stopping if Kl divergence grows too huge... open ai implementation suggestion... if use set default to 0.015
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            print("SPS:", int(self.global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)
            if update % 3 == 0:
                self.save_model(os.sep.join([self.args.model_dir,f"model{update}.pth"]))

        self.envs.close()
        self.writer.close()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), "_".join([path,"actor.pth"]))
        torch.save(self.critic.state_dict(), "_".join([path,"critic.pth"]))

    # for testing only
    def load_actor(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint)
        self.actor.eval()

    # for training only
    def load_model(self, path, layers_to_freeze=[], layers_to_zero=[]):
        actor = torch.load("_".join([path,"actor.pth"]))
        critic = torch.load("_".join([path,"critic.pth"]))
        self.actor.load_state_dict(actor)
        self.critic.load_state_dict(critic)
        self.actor.freeze(layers_to_freeze)
        self.actor.reset(layers_to_zero)
        self.critic.freeze(layers_to_freeze)
        self.critic.reset(layers_to_zero)
