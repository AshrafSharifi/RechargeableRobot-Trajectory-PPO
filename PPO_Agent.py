# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import json
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
from variables import variables
import pandas as pd
from functions import functions
from dB import dB
import pickle

@dataclass
class Args:
    
    reward_temperature_weight:float = 5 # Weight for maximizing temperature change
    
    reward_time_weight:float = 1  # Weight for minimizing time
    
    reward_charging_weight:float = 1
    
    actor_std :float = 0.01
    
    critic_std :float = .9
    
    train_mode : bool = False
    
    train_Forall : bool = False
    
    display_test_results = True
    
    create_train_db = False
    
    threshold_diff: int = 1
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
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
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
    
    state_dim : int = 8
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
        self.actor = self.Actor(x_dim=np.array(envs.state_dim).prod(), actor_layers=[128, 64], activation='relu', u_dim=np.prod(args.action_options-1),std=args.actor_std)
        self.critic = self.Critic(x_dim=np.array(envs.state_dim).prod(), critic_layers=[32, 16, 8], activation='relu',std=args.critic_std)
        

    def get_value(self, x):
        return self.critic(x)


    
    def get_action_and_value(self, x, action=None,best_charging_station=None):
        try:
            logits = self.actor(x)
            if logits is None:
                print("Warning: self.actor(x) returned None.")
                return None, None, None, None
            
            if best_charging_station != None:
               logits[0][best_charging_station-1] = logits.max()+2
                
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


    



def train(args=None,path=None,run_name=None,state_dict = None,train_for_all=False):
    
    retain_graph=False
    CUDA_LAUNCH_BLOCKING=1
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print("batch_size: " + str(args.batch_size))
    print("minibatch_size: " + str(args.minibatch_size))
    print("num_iterations: " + str(args.num_iterations))
    comulative_reward_array = []
    varobj = variables()
    shortest_paths_data = varobj.dikestra()
    with open('data/shortest_paths.json', 'w') as f:
        json.dump(shortest_paths_data, f, indent=4)
        
    varobj.set_shortest_path('data/shortest_paths.json')

    


    
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

    envs = Environment(args.env_id,args.reward_temperature_weight,args.reward_time_weight,args.reward_charging_weight,varobj)
    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
       
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + (envs.state_dim,)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + (envs.action_dim,)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # start_time = time.time()
    
    for iteration in range(1, args.num_iterations):
        
        
        
        
        if iteration > 1 and args.train_Forall:
            next_obs = envs.get_random_state(state_dict)
            his_trajectory,initial_state = envs.extract_history_traj(next_obs)
        else:
            next_obs = envs.reset()
            his_trajectory,initial_state = envs.extract_history_traj(next_obs)
            
        next_obs = torch.Tensor(initial_state).to(device)    
        envs.get_min_max_temp(next_obs)
        next_done = torch.zeros(args.num_envs).to(device)
        # traj_rewards = torch.zeros((args.max_episode_length+1, args.num_envs)).to(device)
        initial_his_trajectory = his_trajectory.copy()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = 0
                while action == 0:
                    bettery_level_temp = next_obs.detach().cpu()[0,6]
                    dist_to_charging_station_temp = next_obs.detach().cpu()[0,7]
                    best_charging_station = None
                    if  bettery_level_temp <= 60 and dist_to_charging_station_temp!=0:
                        # if next_obs[0][0]==4:
                        #     print('ok')
                        best_charging_station = envs.get_best_charging_station(next_obs,his_trajectory)
                    action, logprob, _, value = agent.get_action_and_value(next_obs,None,best_charging_station)

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
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            


            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            # if step >= args.max_episode_length:
            #     next_done = torch.Tensor([1]).to(device) 
            #     comulative_reward = sum(traj_rewards).item()
            #     traj_step = 0
            #     print(f"global_step={global_step}, episodic_return={comulative_reward}")
            #     writer.add_scalar("charts/episodic_return", comulative_reward, global_step)
            #     # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
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
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
                
        

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        
        
        comulative_reward = sum(rewards).item()
        comulative_reward_array.append(comulative_reward)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/comulative_reward", comulative_reward, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        
        
        print("SPS:", int(iteration),"/", int(args.num_iterations),"   Comulative Rewards:", comulative_reward)
       
    
    # Plotting
    # Plotting the cumulative reward
    # plt.plot(comulative_reward_array)
    # plt.xlabel('Episode')
    # plt.ylabel('Cumulative Reward')
    # plt.title('Cumulative Reward over Time')
    # plt.grid(True)
    # plt.show()

    if args.save_model:
        model_path = path+"/model.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    # envs.close()
    writer.close()

def test(args, model_path):
    # Set the random seed for reproducibility
    args.seed = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    varobj = variables()
    varobj.set_shortest_path(os.path.dirname(model_path)+'/shortest_paths.json')
    # Initialize environment
    envs = Environment(args.env_id,args.reward_temperature_weight,args.reward_time_weight,args.reward_charging_weight,varobj)
    envs.get_min_max_temp()
    state = envs.reset()
    state = np.array([state])
    initial_day = int(state[0, 3])
    
    # Initialize agent
    agent = Agent(envs, args).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    # Load the saved model
    # state_dict = torch.load(model_path, map_location=device)
    # agent.load_state_dict(state_dict)
    
    # Set the agent components in evaluation mode
    agent.eval()

   
    varobj.plot_trajectory(None,False)

    
    plot_single_step = False
    temp = envs.get_current_temp(state[0])
    visited_sensor = str(state[0, 0])

    his_trajectory, state = envs.extract_history_traj(state[0])
    his_trajectory[visited_sensor][0] = np.array(state[0])
    his_trajectory[visited_sensor][1] = temp

    finish = False
    num_of_visited_POIs = 0
    time_offset = 15
    complete_path = dict()
    counter = 1
    
    state_counter = 0
    day_counter = 1
    fin_name = 'data/states_dict'
    with open(fin_name, 'rb') as fin:
        states_dict = pickle.load(fin)
    fin.close() 
    states_dict_counts = len(list(states_dict.keys()))    
    day_values = states_dict[str(state[0][1])+'_'+ str(state[0][2])] 
    while not finish:
        
        action, _, _, _ = agent.get_action_and_value(torch.Tensor(state).to(device))
        # Execute the action
        next_state, Flag, reward, temperature_difference, reach_time_minutes, his_trajectory = envs.step(torch.Tensor(state), action, his_trajectory.copy(), 0, time_offset)
        next_state = np.reshape(next_state, [1, envs.state_dim])
        
        prev_battery_level = state[0][6]
        prev_date_time = varobj.extract_date_time(state)

        s_path = varobj.shortest_paths_data[str(state[0][0])][str(next_state[0][0])]['path']
        battery_level = next_state[0][6]
        dist_to_charging_station = next_state[0][7]
        date_time = varobj.extract_date_time(next_state)
        if plot_single_step:
            varobj.plotgraph(s_path[0], s_path, battery_level, round(reward, 2), round(temperature_difference, 2),date_time,dist_to_charging_station)
        
 
        complete_path[counter] = [s_path,[prev_date_time,prev_battery_level],[battery_level,round(reward, 2),date_time,dist_to_charging_station]]
        counter += 1
        
        if Flag:
            if day_counter < len(day_values):
                next_state[0][3] = day_values[day_counter]
                day_counter += 1
            else:
                state_counter += 1
                if state_counter >= states_dict_counts:
                    finish = True
                    break
                year_month = list(states_dict.keys())[state_counter]
                splited_year_month = year_month.split('_')
                next_state[0][1] = int(splited_year_month[0])
                next_state[0][2] = int(splited_year_month[1])
                day_values = states_dict[str(next_state[0][1])+'_'+ str(next_state[0][2])]
                next_state[0][3] = day_values[0]
                day_counter = 1
                # next_state[0][4] = 0
                # next_state[0][3] = 0
  
        state = next_state
    # Save dictionary to a file
    
    with open(os.path.dirname(model_path)+'/complete_path.pkl', 'wb') as file:
        pickle.dump(complete_path, file)
    filtered_items = {k: v for k, v in complete_path.items() if v[1][1] == varobj.initialChargingLevel}
    print("The number of stoppings for getting charge: " + str(len(filtered_items)-1))
    if args.display_test_results:
        for item in complete_path.values():
            traj = dict()
            traj[0]= item
            varobj.plot_trajectory(traj)            
    if args.create_train_db:
        db = dB(varobj)
        db.create_train_db(complete_path)
        

if __name__ == "__main__":
    args = tyro.cli(Args)
    KMP_DUPLICATE_LIB_OK= True
    if args.train_Forall:
        fin_name = 'data/states_dict'
        with open(fin_name, 'rb') as fin:
            states_dict = pickle.load(fin)
        fin.close() 
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%H%M")
        run_name = f"{formatted_datetime}"
        path = f"data/Runs/ChangedEnvironmentConfiguration_{run_name}"
        train(args,path,run_name,states_dict,True)
    elif args.train_mode:
        
        # for args.actor_std in [.5]:
        #     for args.critic_std in [.5]:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%H%M")
        run_name = f"{formatted_datetime}"
        path = f"data/Runs/{run_name}"
        print(args.actor_std,args.critic_std)
        train(args,path,run_name)
    else:
        model_path = 'data/Runs/ChangedEnvironmentConfiguration_1541/model.cleanrl_model'
        test(args,model_path)


            
    