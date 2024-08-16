# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from Environment import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime

@dataclass
class Args:
    
    train_mode : bool = True
    
    train_Forall : bool = False
    
    threshold_diff: int =1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: object = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Field_Temp"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 26
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.00
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: object = None
    """the target KL divergence threshold"""
    
    save_model : bool = True
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    state_dim : int = 6
    """Dimension of the state"""
    action_options : int =7
    """Number of possible actions"""
    
    max_episode_length: int = 12


def make_env(env_id, idx, capture_video, run_name):
    env = Environment(env_id)
    return env


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def check_weights_for_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            param.data[param.isnan()] = 0.000001
            print(f"Found NaN values in the weights of layer {name}")
        elif (param == 0).any():
            param.data[param.data == 0] = 0.001
            # param.data.add_(0.001)
            print(f"All weights are zero in layer {name}")


class Agent(nn.Module):
    
    def Actor(self, x_dim, actor_layers=None, activation='relu', u_dim=1, std= .5):
        if actor_layers is None:
            actor_layers = [6]
        layers = []
        layers.append(layer_init(nn.Linear(np.array(args.state_dim).prod(), actor_layers[0])))
        layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
        for i in range(1, len(actor_layers)):
            layers.append(layer_init(nn.Linear(actor_layers[i - 1], actor_layers[i])))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
        layers.append(layer_init(nn.Linear(actor_layers[-1], args.action_options), std=std))
        return nn.Sequential(*layers)

    def Critic(self, x_dim, critic_layers=None, activation='relu', std=0.01):
        if critic_layers is None:
            critic_layers = [10, 8]
        layers = []
        layers.append(layer_init(nn.Linear(x_dim, critic_layers[0])))
        layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
        for i in range(1, len(critic_layers)):
            layers.append(layer_init(nn.Linear(critic_layers[i - 1], critic_layers[i])))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
        layers.append(layer_init(nn.Linear(critic_layers[-1], 1), std=std))
        return nn.Sequential(*layers)
    
    def __init__(self, envs, args):
        super().__init__()
        self.actor = self.Actor(x_dim=np.array(envs.state_dim).prod(), actor_layers=[32, 32], activation='tanh', u_dim=np.prod(args.action_options-1),std=.5)
        self.critic = self.Critic(x_dim=np.array(envs.state_dim).prod(), critic_layers=[32, 16, 8], activation='tanh',std=0.01)
        
        
            
   
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(args.state_dim).prod(), 32)),
        #     nn.Relu(),
        #     layer_init(nn.Linear(32, 32)),
        #     nn.Relu(),
        #     nn.Linear(32, args.action_options),  # Remove layer_init for the last layer
        #     nn.Softmax(dim=-1)  # Apply Softmax activation
        # )
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(args.state_dim).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     nn.Linear(64, args.action_options),  # Remove layer_init for the last layer
        #     nn.Softmax(dim=-1)  # Apply Softmax activation
        # )

    def get_value(self, x):
        return self.critic(x)

    

    def get_action_and_value__(self, x, action=None):
        try:
            # check_weights_for_nan(agent) 
            if np.random.rand() <= self.epsilon:
                action = torch.randint(1, 8, (1,)).item()
                random_initial_steps -= 1
                log_prob = torch.tensor(0.0)  # Set a default value for log probability
                entropy = torch.tensor(0.0)  # Set a default value for entropy
            else:
                # check_weights_for_nan(agent) 
                logits = self.actor(x)
                if logits is None:
                    print("Warning: self.actor(x) returned None.")
                    return None, None, None, None
        
                probs = Categorical(logits=logits)
                if action is None:
                    action = probs.sample()+1
                    while int(x[0, 0].item()) == action or action==0:
                        action = probs.sample()+1
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None
    
    def get_action_and_value(self, x, action=None):
        try:
            logits = self.actor(x)
            if logits is None:
                print("Warning: self.actor(x) returned None.")
                return None, None, None, None
        
            probs = Categorical(logits=logits)
            
            if action is None:
                # Sample an action
                action = probs.sample()
    
            # Ensure all sampled actions are valid for your environment
            while int(x[0][0])== int(action[0]+1) or not torch.all((action >= 0) & (action< 7)):  # Assuming 6 categories (0 to 5)
                action = probs.sample()
    
            # Adjust the action values by 1 to match your state numbering
            adjusted_action = action + 1
    
            # Compute log probability and entropy for the adjusted actions
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()
    
            return adjusted_action, log_prob, entropy, self.critic(x)
        
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None




    # def get_action_and_value(self, x, action=None):
    #     try:
    #         # check_weights_for_nan(agent) 
    #         logits = self.actor(x)
    #         if logits is None:
    #             print("Warning: self.actor(x) returned None.")
    #             return None, None, None, None
    
    #         probs = Categorical(logits=logits)
    #         if action is None:
    #             action = probs.sample()
    #             while int(x[0, 0].item()) == action+1 or action==0:
    #                 action = probs.sample()
    #         return action+1, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return None, None, None, None
        
        
    # def get_action_and_value(self, x, action=None):
    #     try:
    #         logits = self.actor(x)
    #         probs = F.softmax(logits, dim=-1)
    
    #         if action is None:
    #             action = Categorical(probs=probs).sample() + 1  # Adding 1 to make the range 1 to 6
    
    #         if action.dim() == 0:
    #             action = action.unsqueeze(0)  # Convert scalar to 1D tensor
    
    #         # Reshape logits to have two dimensions
    #         logits_reshaped = logits.view(1, -1)
    
    #         log_prob = F.log_softmax(logits_reshaped, dim=1).gather(1, action.unsqueeze(1))
    
    #         entropy = -(probs * F.log_softmax(logits_reshaped, dim=1)).sum(-1)
    
    #         return action, log_prob, entropy, self.critic(x)
    
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return None, None, None, None


def train(args=None,path=None,state_dict = None,train_for_all=False):
    
    retain_graph=False
    CUDA_LAUNCH_BLOCKING=1
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print("batch_size: " + str(args.batch_size))
    print("minibatch_size: " + str(args.minibatch_size))
    print("num_iterations: " + str(args.num_iterations))

    
    writer = SummaryWriter(path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = Environment(args.env_id)
    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)
       
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (envs.state_dim,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (envs.action_dim,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    for iteration in range(1, args.num_iterations):
        
        
        
        
        if iteration > 1 and train_for_all:
            next_obs = [0] * 6
            next_obs[0] =random.choice(range(1,8))
            keys = list(state_dict.keys())
            random_key = random.choice(keys)
            [y,m] = random_key.split('_')
            next_obs[1]=int(y)
            next_obs[2]=int(m)
            next_obs[3] = random.choice(state_dict[random_key])
            random_hour = random.randint(0, 23)
            random_min = random.choice([0, 15, 30, 45])
            next_obs[4] = random_hour
            next_obs[5] = random_min
            his_trajectory,initial_state = envs.extract_history_traj(next_obs)
        else:
            next_obs = envs.reset()
            his_trajectory,initial_state = envs.extract_history_traj(next_obs)
            
        next_obs = torch.Tensor(initial_state).to(device)    
        envs.get_min_max_temp(next_obs)
        next_done = torch.zeros(args.num_envs).to(device)
        traj_rewards = torch.zeros((args.max_episode_length+1, args.num_envs)).to(device)
        initial_his_trajectory = his_trajectory.copy()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        traj_step = 0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = 0
                while action == 0:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            
            # set choosed sensor as next step
            next_obs,flag,reward,temperature_difference,reach_time_minutes,his_trajectory = envs.step(next_obs,action,his_trajectory)  # Update the state based on the robot's movement
            next_obs = torch.Tensor([next_obs]).to(device) 
            next_done = torch.zeros(1).to(device) 
            # next_done = torch.Tensor(0).to(device) 
            # reward,temperature_difference,reach_time_minutes,his_trajectory = dqn.calculate_reward(state, next_state,his_trajectory)  # Calculate the reward
            
            
            # next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # next_done = np.logical_or(terminations, truncations)
            traj_rewards[traj_step] = reward
            traj_step += 1
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            if step >= args.max_episode_length:
                next_done = torch.Tensor([1]).to(device) 
                comulative_reward = sum(traj_rewards).item()
                traj_step = 0
                print(f"global_step={global_step}, episodic_return={comulative_reward}")
                writer.add_scalar("charts/episodic_return", comulative_reward, global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps-1)):
                if t == args.max_episode_length:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (envs.state_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (envs.action_dim,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # print(pg_loss1,pg_loss2,pg_loss)
                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                    # print(v_loss)
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    # print(v_loss)

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                # print(loss)
                if loss > 1e9 or torch.isnan(loss).any():
                    print(loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(iteration),"/", int(args.num_iterations))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
    if args.save_model:
        model_path = path+f"/{args.exp_name}.cleanrl_model"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    # envs.close()
    writer.close()
    

def test(args,model_path):
    args.seed = 1
    envs = Environment(args.env_id)
    envs.get_min_max_temp()
    state = envs.reset()
    state = np.array([state])
    initial_day = int(state[0,3])
    agent = Agent(envs, args)
    his_trajectory,state = envs.extract_history_traj(state[0])
    state_dict = torch.load(model_path)
    
    # Load the state dictionary components separately
    # agent.actor.model.load_state_dict({k.replace('actor.model.', ''): v for k, v in state_dict.items() if 'actor' in k})
    
    # Set the agent components in evaluation mode
    agent.actor.eval()
  


    loaded_model=agent.actor
    
    
    path = {
        "1": ["NotVisited", 1, 0, 0, 0, np.empty((0, 0)),0],    #POI number: ['status','priority(based on article)'
        "2": ["NotVisited", 2, 0, 0, 0, np.empty((0, 0)),0],    #             'reward','temp difference', 'time to reach', state
        "4": ["NotVisited", 3, 0, 0, 0, np.empty((0, 0)),0],    #             'temperature']
        "6": ["NotVisited", 4, 0, 0, 0, np.empty((0, 0)),0],
        "7": ["NotVisited", 5, 0, 0, 0, np.empty((0, 0)),0],
        "5": ["NotVisited", 6, 0, 0, 0, np.empty((0, 0)),0],
        "3": ["NotVisited", 7, 0, 0, 0, np.empty((0, 0)),0]
           }
    
    
    # flag for reward of first item
    control_flag = False
    # compute reward & temperature difference for the k first items of path
    last_elemnt_his = list(his_trajectory.items())[-1][1][0]
    first_three_items = {k: his_trajectory[k] for i, k in enumerate(his_trajectory) if i < 2}
    temp_rewards= dict()
    for key, value in first_three_items.items():
        temp_st = value[0]
        temp_new_state, temp_Flag,temp_reward,temp_temperature_difference,temp_reach_time_minutes,temp_his_trajectory = envs.step(torch.Tensor([last_elemnt_his]),value[0][0],his_trajectory.copy())
        temp_rewards[key] = [temp_reward,temp_temperature_difference,temp_new_state,temp_temperature_difference,temp_reach_time_minutes]
        if temp_temperature_difference > args.threshold_diff:
            break

    

    max_index, max_item = max(enumerate(temp_rewards.items()), key=lambda x: x[1][1][0])
    # max_item ('4', [6.3449999999999935, 1.3739999999999988, [4, 2021, 8, 27, 2, 15], 1.3739999999999988, 45])
    temp_initial = state.copy()
    state = np.array([max_item[1][2]])
    temp = envs.get_current_temp(state[0])
    visited_sensor = str(state[0,0])
    path[visited_sensor][0] = 'Visited' 
    path[visited_sensor][2] =  max_item[1][0] #reward
    path[visited_sensor][3] =  max_item[1][3] #temp difference
    path[visited_sensor][4] = max_item[1][4]  #time to reach
    path[visited_sensor][5] = np.array([max_item[1][2]]) #state
    path[visited_sensor][6] = temp #temperature

    his_trajectory[visited_sensor][0]=np.array(state[0])
    his_trajectory[visited_sensor][1]=temp
    
    passed_key = list()
    for key,value in temp_rewards.items():
        if path[key][1] < path[visited_sensor][1]:
            path[key][0] = 'Passed'
            path[key][2] =  value[0] #reward
            path[key][3] =  value[3] #temp difference
            path[key][4] = value[4]  #time to reach
            path[key][5] = np.array([value[2]]) #state
            path[key][6] = envs.get_current_temp(path[key][5][0]) #temperature
            

    finish = False
    num_of_visited_POIs = 0
    keys = []
    while finish == False:
        # Reshape the new state if neededhisory_trajectory=his_trajectory
        temp = loaded_model(torch.Tensor(state))
        temp = temp.detach().numpy()[0]
        path_nodes = dict()
        POI = 1
        for item in temp:
            path_nodes[str(POI)] = [path[str(POI)][0],item]
            POI += 1
            
        max_value = float('-inf')
        max_item = None  
        for key, value in path_nodes.items():
            if key != state[0,0] and value[0] == 'NotVisited' and value[1] > max_value:
                max_value = value[1]
                max_item = (key, value)

        action = int(max_item[0])        
        path[str(action)][0] = 'Visited' 
        
        # Number of visited items before action 
        v_keys= [key for key, value in path.items() if value[1] < path[str(action)][1] and value[0] == 'Visited']
        time_offset = len(v_keys) * envs.process_time
        # Set the POI which are located befor the current state and not selected by algorithm to passed
        passedkeys = list()
        current_key = path[str(state[0][0])][1]+1
        end_key = path[str(action)][1]
        while current_key != end_key:
            if(int(current_key)==8):
                current_key=1
            p_k = [key for key, value in path.items() if value[1] == int(current_key) and value[0] == 'NotVisited']
            if len(p_k)!=0:
                passedkeys.append(p_k)
                path[p_k[0]][0] = 'Passed'
                
                p_next_state, p_Flag,p_reward,p_temperature_difference,p_reach_time_minutes,_ = envs.step(torch.Tensor(state),int(p_k[0]),his_trajectory.copy(),0,time_offset) 
                
                
                path[p_k[0]][2] =  p_reward #reward
                path[p_k[0]][3] =  p_temperature_difference #temp difference
                path[p_k[0]][4] = p_reach_time_minutes  #time to reach
                path[p_k[0]][5] = np.array([p_next_state]) #state
                path[p_k[0]][6] = envs.get_current_temp(p_next_state) #temperature
                next_key = path[p_k[0]][1]+1
                current_key = int(next_key)
            else:
                break
            
      
      
        
        p_key = envs.get_previous_key(path, str(action))
        is_visited=1
        next_state, Flag,reward,temperature_difference,reach_time_minutes,his_trajectory = envs.step(torch.Tensor(state),action,his_trajectory.copy(),0,time_offset) 
        
        next_state = np.reshape(next_state, [1, envs.state_dim])
        

            
        path[str(action)][2] = reward
        path[str(action)][3] = temperature_difference
        path[str(action)][4] = reach_time_minutes 
        path[str(action)][5] =  next_state
        
        if len(keys)==1:
            print('1')
        if temperature_difference < args.threshold_diff and len(keys)==1:
            path[str(action)][0] = 'Passed' 
        else:
            state = next_state
        keys = {key for key, value in path.items() if value[0] == 'NotVisited'}
        # if len(keys) == 0 or int(state[0][4]) == 0:
        if len(keys) == 0 or Flag:
            finish = True
            if Flag:
                state[0][3]=state[0][3]+1
                path[str(action)][5] =  state
        
            
             
         
    
    table = envs.print_traj(path,True,his_trajectory)
    
    
    last_state = (table[-1][6])
    # if path[last_state[0,0]]=='Visited':
    current_sensor = last_state[0,0]
    current_hour = last_state[0,4]
    current_minute = last_state[0,5]
    base_time = str(current_hour)+':'+str(current_minute)+':00'
    next_hour, next_min, Flag = envs.func.add_minutes(base_time, envs.process_time+15 )
    
    next_state = envs.find_next_item(last_state[0,0])
    
    temp_var = state[0,3].copy()
    if Flag:
        temp_var=state[0,3]+1
        
    new_state = [next_state,state[0,1],state[0,2],temp_var,next_hour,next_min]
        
    
    return np.reshape(new_state, [1, envs.state_dim]),path,his_trajectory,table[-1][6],table[0][6]

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.train_Forall:
        fin_name = 'data/states_dict'
        with open(fin_name, 'rb') as fin:
            states_dict = pickle.load(fin)
        fin.close() 
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d__%H-%M")
        run_name = f"total_train__{args.env_id}__{args.exp_name}__{formatted_datetime}"
        path = f"runs/{run_name}"
        train(args,path,states_dict,True)
    elif args.train_mode:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d__%H-%M")
        run_name = f"{args.env_id}__{args.exp_name}__{formatted_datetime}"
        path = f"runs/{run_name}"
        train(args,path)
    else:
        model_path = 'runs/total_train__Field_Temp__ppo_agent__2024_02_27__23-51/ppo_agent.cleanrl_model'
        test(args,model_path)
    