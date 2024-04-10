
"""
# Modified partial online planning using Thompson sampling from Pascal Poupart's lecture on BRL
# Modelling uncertainty over both the transition model and reward function
            # Note: Transition model is tabular but can be replaced with function approximator 
            #       However, this would result in the distributions changing to normal - gamma i
"""

from scipy.stats import beta
import numpy as np
from tqdm import tqdm
from .env import priors, update_prior_transition, update_prior_reward
from .mdp import MDP


def play_episode(Q, env, max_steps_per_episode, seed):  
    
    state = env.reset(seed=seed)[0]
    r = 0
    all_states = []
    all_actions = []
    all_states.append(state)
    for step in range(max_steps_per_episode):  
        
        #action = argmaxrand(Q[:,state])     
        action = np.argmax(Q[:,state])  
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        all_states.append(new_state)
        all_actions.append(action)
        # r = np.where(new_state==7, 100,0)
        r += reward
            
        if done:            
            return r, step+1, all_states, all_actions  
                    
        else:  
            state = new_state    
    
    return r, step+1, all_states, all_actions
            

class BRLAgent():
    
    def __init__(self, num_episodes, nt, max_steps_per_episode, env, seed, a_t=1, b_t=1, a_r=1, b_r=1, k=32, render=False):
            
            # Trial parametres:
            self.max_steps_per_episode = max_steps_per_episode 
            self.nt = nt
            self.num_episodes = num_episodes
            self.k = k
            self.render = render
            self.seed = seed
            
            # Environment set-up:
            self.env = env
            
            # Belief state hyperparameter:
            self.a_t, self.b_t, self.a_r, self.b_r = a_t, b_t, a_r, b_r
            self.belief_states = [(a_t,b_t,a_r,b_r)]   
    
            # Saving score:
            self.tr = np.zeros(self.num_episodes)
            self.ts = np.zeros(self.num_episodes)         
            
            self.tr_online = np.zeros(self.num_episodes)
            self.ts_online = np.zeros(self.num_episodes)

            self.all_states = []
            self.all_actions = []         
        
        
            
    def planner(self): 
        
            # Planning bayesian optimal behaviour based on thompson sampling approximation
            
            self.state = self.env.reset(seed=self.seed)[0]
            done = False
            
            for i in range(self.nt):
                    
                    # sample from k thetas from belief (i.e. beta) distribution for transition & reward model model:          
                    theta = zip(beta.rvs(self.a_t,self.b_t,size=self.k), beta.rvs(self.a_r, self.b_r, size=self.k))
                    
                    # create k MDP using the sampled thetas
                    MDPs = [priors(t_t, t_r, self.env, 0.9) for t_t, t_r in theta] 
                    
                    # solve the k MDPs using value iteration.
                    Q_functions = [m.valueiteration(np.zeros(self.env.observation_space.n)) for m in MDPs]        
                    
                    # average Q-value 
                    Q_hat = np.mean(Q_functions,axis=0)
                    
                    # get action to play via max(a) Q-hat(s,a):
                    # action = ut.argmaxrand(Q_hat[:,self.state])
                    
                    action = np.argmax(Q_hat[:,self.state])
                    # sample next state:            
                    new_state, reward, terminated, truncated, info = self.env.step(action) 
                    done = terminated or truncated
                    # update priors:
                    self.a_t,self.b_t = update_prior_transition(self.a_t,self.b_t,action,self.state,new_state, 1)
                    self.a_r,self.b_r = update_prior_reward(self.a_r,self.b_r,reward,new_state)
                        
                    self.belief_states.append((self.a_t,self.b_t,self.a_r,self.b_r)) # storing the beta distribution hyper-parametres   
                    
                    if done:
                        break
                    
                    self.state = new_state           
        
            return Q_hat, self.belief_states, reward, i

    def simulator(self):
            
            # Acting bayes optimal
            
            for episode in tqdm(range(self.num_episodes)):
                # random.seed(episode)
                
                # self.env.seed(episode)
                
                Q, self.belief_states, reward,i = self.planner()                     
              
                # Updating priors:    
                self.a_t = self.belief_states[-1][0]
                self.b_t = self.belief_states[-1][1]
                
                self.a_r = self.belief_states[-1][2]
                self.b_r = self.belief_states[-1][3]
                                        
                # Storing score from play:  
                self.tr_online[episode], self.ts_online[episode] = reward, i+1
                self.tr[episode], self.ts[episode], \
                all_states, all_actions = play_episode(Q, self.env, self.max_steps_per_episode, self.seed)
                self.all_states += all_states
                self.all_actions += all_actions
                
            return self.tr, self.ts, Q, self.all_states, self.all_actions
