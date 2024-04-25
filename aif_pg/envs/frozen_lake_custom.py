import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns

from pymdp.envs import Env
from pymdp import utils, maths


LOCATION_FACTOR_ID = 0
TRIAL_FACTOR_ID = 1

LOCATION_MODALITY_ID = 0
REWARD_MODALITY_ID = 1

REWARD_IDX = 1
LOSS_IDX = 2

# ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

class FrozenLake_Custom(Env):
    """ States:
            Location - All the individual locations of the grid ( Eg. 1x9 vector for 3x3 grid)
            Context - To specify the hole and goal interchanging (1x2 vector)
        
        Obs:
            Location - Identical to location hidden state (Eg. 9 categorical location one-hots for 3x3 grid)
            Reward - [No_reward, Reward, Loss]
    """
    
    def __init__(self, reward_condition=1):
        self.grid_dims = [3, 3]
        self.num_locations = np.prod(self.grid_dims)
        self.num_reward_conditions = 2
        self.num_states = [self.num_locations, self.num_reward_conditions]
        self.num_controls = [len(ACTIONS), 1]
        self.num_obs = [self.num_locations, self.num_reward_conditions + 1]
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)


        # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates 
        grid = np.arange(self.num_locations).reshape(self.grid_dims)
        it = np.nditer(grid, flags=["multi_index"])

        self.loc_list = []
        while not it.finished:
            self.loc_list.append(it.multi_index)
            it.iternext()

        (self.reward_loc, self.hole_loc) = self.set_reward_locs(reward_condition)
        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = reward_condition
        self._state = None
    
    def reset(self, state=None, reward_condition=None):
        if state is None:
            loc_state = utils.onehot(0, self.num_locations)
            
            # self._reward_condition = np.random.randint(self.num_reward_conditions)
            self._reward_condition = reward_condition if reward_condition is not None \
                                      else self._reward_condition
                 
            reward_condition_categorical = utils.onehot(self._reward_condition, self.num_reward_conditions)
            # TODO: Should the A and B matrices change??
            # Maybe not. since we are encoding the context already. Unless the locations of them are switched
            
            full_state = utils.obj_array(self.num_factors)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition_categorical
            self._state = full_state
        else:
            self._state = state
        return self._get_observation()
    
    def reset_env(self, reward_condition):
        return self.reset(state=None,reward_condition=reward_condition)
    
    def set_reward_locs(self,reward_condition=1):
        (reward_loc, hole_loc) = (8, 6) if reward_condition==1 else (6, 8)
        return (reward_loc, hole_loc)

    def step(self, actions):
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        return self._get_observation()
    
    def render(self):
        pass

    def sample_action(self):
        return [np.random.randint(self.num_controls[i]) for i in range(self.num_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist

    def get_transition_dist(self):
        return self._transition_dist


    def get_rand_likelihood_dist(self):
        pass

    def get_rand_transition_dist(self):
        pass

    def _get_observation(self):

        prob_obs = [maths.spm_dot(A_m, self._state) for A_m in self._likelihood_dist]

        obs = [utils.sample(po_i) for po_i in prob_obs]
        return obs

    def _construct_transition_dist(self):

        # initialize the shapes of each sub-array `B[f]`
        B_f_shapes = [ [ns, ns, self.num_controls[f]] for f, ns in enumerate(self.num_states)]

        # create the `B` array and fill it out
        B = utils.obj_array_zeros(B_f_shapes)

        # fill out `B[0]` using the 
        for action_id, action_label in enumerate(ACTIONS):

            for curr_state, grid_location in enumerate(self.loc_list):

                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y 
                    next_x = x
                elif action_label == "DOWN":
                    next_y = y + 1 if y < (self.grid_dims[0]-1) else y 
                    next_x = x
                elif action_label == "LEFT":
                    next_x = x - 1 if x > 0 else x 
                    next_y = y
                elif action_label == "RIGHT":
                    next_x = x + 1 if x < (self.grid_dims[1]-1) else x 
                    next_y = y
                elif action_label == "STAY":
                    next_x = x
                    next_y = y

                new_location = (next_y, next_x)
                next_state = self.loc_list.index(new_location)
                B[LOCATION_FACTOR_ID][next_state, curr_state, action_id] = 1.0
        
        B[TRIAL_FACTOR_ID][:,:,0] = np.eye(self.num_states[1])

        return B

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([[obs_dim] + self.num_states for obs_dim in self.num_obs])
        
        # make the location observation only depend on the location state (proprioceptive observation modality)
        A[LOCATION_MODALITY_ID] = np.tile(np.expand_dims(np.eye(self.num_locations), (-1)), (1, 1, self.num_states[1]))

        ## Reward addition
        # make the reward observation depend on the location (being at reward location) and the reward condition
        A[REWARD_MODALITY_ID][0,:,:] = 1.0  # default makes Null the most likely observation everywhere.
                                            # 1 for almost all the locations with 'No_reward' obs
        
        reward_loc_idx = self.reward_loc
        hole_loc_idx = self.hole_loc

        # Setting appropriate values for the reward location
        A[REWARD_MODALITY_ID][0,reward_loc_idx,:] = 0.0
        A[REWARD_MODALITY_ID][1,reward_loc_idx,0] = 1.0
        A[REWARD_MODALITY_ID][2,reward_loc_idx,1] = 1.0

        # Setting appropriate values for the hole location
        A[REWARD_MODALITY_ID][0,hole_loc_idx,:] = 0.0
        A[REWARD_MODALITY_ID][1,hole_loc_idx,1] = 1.0
        A[REWARD_MODALITY_ID][2,hole_loc_idx,0] = 1.0

        return A

    def _construct_state(self, state_tuple):

        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)

        return state

    @property
    def state(self):
        return self._state

    @property
    def reward_condition(self):
        return self._reward_condition