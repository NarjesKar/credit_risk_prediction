from collections import deque
import numpy as np
from bisect import bisect

class best_experiences:
    def __init__(self,var_list = ['s','s_','a','r','best_return'],max_length = 5,init_best = -100000):
        self.max_length = max_length
        self.var_list = var_list
        self.mem = {}
        for v in var_list:
            self.mem[v] = deque()
        self.mem['best_return'].append(init_best)
    def update_on_rule(self,dict_to_inject,episode_return):
        if(len(self.mem['best_return']) <= self.max_length):
            if(episode_return > max(self.mem['best_return'])):
                for v in self.var_list[:-1]: 
                    self.mem[v].append(dict_to_inject[v]) 
                # make updates
                self.mem['best_return'].append(episode_return)
                # update the best reward
        else:
            # update the value in the middle
            if(episode_return > min(self.mem['best_return']) and episode_return < max(self.mem['best_return'])):
                print("we are here")
                pos = bisect(self.mem['best_return'],episode_return)
                for v in self.var_list[:-1]: 
                    self.mem[v][pos - 1] = dict_to_inject[v]
                self.mem['best_return'][pos - 1] = episode_return
            # update the last value
            elif(episode_return >= max(self.mem['best_return'])):
                for v in self.var_list[:-1]: 
                    self.mem[v][-1] = dict_to_inject[v]
                self.mem['best_return'][-1] = episode_return
            
