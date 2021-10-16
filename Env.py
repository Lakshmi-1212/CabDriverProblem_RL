# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

waiting_time = 1 # Waiting time to be added if driver does not pick a ride
terminal_hours = 24*30

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [(xi,tj,dk) for xi in range(m) for tj in range(t) for dk in range(d)]
        #self.state_init = random.choice(self.state_space)
        
        # Track the elapsed hours to check if terminal state is reached
        self.elapsed_hours = 0

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input - Architecture-1

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # state = (current location, time, day of the week) -> 5 bits + 24 bits + 7 bits = 36 bits
        if not state:
            return
        
        state_encod = [0] * (m + t + d)
        
        # encode location
        state_encod[state[0] - 1] = 1
        
        # encode hour of the day
        state_encod[m + state[1]] = 1
        
        # encode day of the week
        state_encod[m + t + state[2]] = 1
        
        return state_encod
    
    
     


    # Use this function if you are using architecture-2 
#     def state_encod_arch2(self, state, action):
#         """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
#         return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_index,actions   
    

    


    def get_updated_time_day(self, current_time, current_day, additional_time):
        new_time = current_time + additional_time
        new_day = current_day
        if new_time>=24:
            new_time -= 24
            new_day = 0 if (current_day==6) else current_day+1
            
        return int(new_time), int(new_day)
        
        
    def is_terminal(self):
        # Terminal state is reached when the driver completes 24*30 hours 
        return True if self.elapsed_time >= terminal_hours else False


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        xi = state[0]
        tj = state[1]
        dk = state[2]
        p = action[0]
        q = action[1]
        
        if ((p==q) and (p==0)):
            # reward for no-ride action 
            reward = -C
        else:
            # Compute the reward based on the pickup and drop location
            time_i_p = Time_matrix[xi][p][tj][dk]
            time_updated, day_updated = self.get_updated_time_day(tj, dk, time_i_p)
            time_p_q = Time_matrix[p][q][time_updated][day_updated]
            
            reward = R * time_p_q - C * (time_i_p + time_p_q)
            
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        xi = state[0]
        tj = state[1]
        dk = state[2]
        p = action[0]
        q = action[1]
        trip_time = 0
        if ((p==q) and (p==0)):
            # if driver is not accepting a ride (no-ride), location is the same, waiting time needs to be added
            new_location = xi
            new_time, new_day = self.get_updated_time_day(tj, dk, waiting_time)
            trip_time = waiting_time
        else:
            # update location and time for new pickup and drop locations
            new_location = q
            time_i_p = Time_matrix[xi][p][tj][dk]
            time_updated, day_updated = self.get_updated_time_day(tj, dk, time_i_p)
            
            time_p_q = Time_matrix[p][q][time_updated][day_updated]
            new_time, new_day = self.get_updated_time_day(time_updated, day_updated, time_p_q)
            
            
            trip_time = time_i_p + time_p_q
            
        new_state = (new_location,new_time,new_day)
        
        self.elapsed_time += trip_time
         
        #print(f'4DEBUG - trip:({state},{action},{reward},{new_state}), time:{self.elapsed_time},{episode_done}')
            
        return new_state



    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        # Get the next state for the given state & action
        new_state = self.next_state_func(state, action, Time_matrix)
        

        # Get the reward for the given state and action
        reward = self.reward_func(state, action, Time_matrix)
         
        # Check if terminal condition is reached
        episode_done = self.is_terminal()
    
    
        return new_state, reward, episode_done
    
    

    def reset(self):
        #print(f'RESETTING')
         
        self.state_init = random.choice([(0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0)])
        self.elapsed_time = 0
        return self.state_init
