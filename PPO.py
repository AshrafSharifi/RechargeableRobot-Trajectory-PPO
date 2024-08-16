import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import pickle

class PPO:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=3e-4, update_interval=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_interval = update_interval
        
        # Policy and value networks
        self.policy_network = self.build_policy_network()
        self.value_network = self.build_value_network()
        
        # Optimizers
        self.policy_optimizer = Adam(learning_rate=lr)
        self.value_optimizer = Adam(learning_rate=lr)
        
        # Replay buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
    
    def build_policy_network(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='softmax')
        ])
        return model
    
    def build_value_network(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        return model
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_dim])
        probs = self.policy_network(state)
        action = np.random.choice(self.action_dim, p=np.squeeze(probs))
        log_prob = np.log(probs[0, action])
        return action, log_prob
    
    def remember(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def discount_rewards(self, rewards, dones):
        discounted_rewards = []
        cumulative = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                cumulative = 0
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        return discounted_rewards
    
    def get_min_max_temp(self,state=None):
        # temp_data =  self.sensorsdata["sensor"+str(state[0])]
        # temp_data = [value for value in temp_data.values()][0]
        # state_data = [value for value in temp_data.values() if value['year'] == state[1] and value['month'] == state[2] and value['day'] == state[3]] 
        # temp_var = state_data[0]['sensorData']
        # self.min_temp = min(temp_var["temp_to_estimate"])
        # self.max_temp = max(temp_var["temp_to_estimate"])
        
        fin_name = 'data/temp_info' 
        with open(fin_name, 'rb') as fin:
            temp_info = pickle.load(fin)
        fin.close() 
        
        self.min_temp = temp_info['min_temp']
        self.max_temp = temp_info['max_temp'] 
    
    
    def update_policy(self):
        
        
    
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = self.discount_rewards(self.rewards, self.dones)
        rewards = np.array(rewards)
        log_probs_old = np.array(self.log_probs)

        with tf.GradientTape() as tape:
            log_probs = self.policy_network(states, training=True)
            log_probs = tf.reduce_sum(log_probs * tf.one_hot(actions, self.action_dim), axis=1)
            ratio = tf.exp(log_probs - log_probs_old)
            surrogate1 = ratio * rewards
            surrogate2 = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * rewards
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

        policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))

        with tf.GradientTape() as tape:
            values = self.value_network(states, training=True)
            value_loss = tf.reduce_mean(tf.square(rewards - values))

        value_grads = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))
    
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def find_next_item(self,arr, num):
        
    
        # Find the index of the given number in the array
        try:
            index = arr.index(num)
        except ValueError:
            print(f"The number {num} is not in the array.")
            return None
    
        # Calculate the index of the next number in the circular array
        next_index = (index + 1) % len(arr)
    
        # Return the next number
        return arr[next_index]  
    
    def extract_history_traj(self,dqn,current_state):
        current_sensor = current_state[0,0]
        next_sensor=0
        path = [1,2,4,6,7,5,3]
        history_path = dict()
        initial_sensor=current_state[0,0]
    
        while initial_sensor != next_sensor:
            current_state = np.reshape(current_state, [1, dqn.state_dim])
            current_sensor = current_state[0,0]
            temperature = dqn.get_current_temp(current_state)
            history_path[str(current_sensor)]=[current_state, temperature]
            base_time = str(current_state[0,4]) + ':' + str(current_state[0,5]) + ':00'
            next_hour, next_min, Flag = dqn.func.add_minutes(base_time, 15)
            next_sensor = self.find_next_item(path,current_state[0,0])
            current_state = np.array([int(next_sensor), int(current_state[0,1]), int(current_state[0,2]), int(current_state[0,3]), next_hour, next_min])
        
        # Custom sorting key based on day, hour, and minutes
        def sorting_key(item):
            return tuple(item[0][0, 2:5])
        
        # Sort the dictionary based on the custom key
        his_trajectory = dict(sorted(history_path.items(), key=lambda x: sorting_key(x[1])))
        last_item = list(his_trajectory.values())[-1]
        current_state = last_item[0]
        base_time = str(current_state[0,4]) + ':' + str(current_state[0,5]) + ':00'
        next_hour, next_min, Flag = dqn.func.add_minutes(base_time, 15)
        next_sensor = self.find_next_item(path,current_state[0,0])
        current_state = np.array([int(next_sensor), int(current_state[0,1]), int(current_state[0,2]), int(current_state[0,3]), next_hour, next_min])
    
        return history_path,np.reshape(current_state, [1, dqn.state_dim])
    
 
        
        



