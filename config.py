import tensorflow as tf

class config_cartpole:
    def __init__(self):
        # environment specific config
        self.d2v = False
        self.rendering = False
        self.use_optimal_baseline = False
        # Timescale parameters
        self.lr_timescale = 1.0
        self.step_timescale = 1
        self.lr_actor = 0.01
        # Training parameters
        self.number_of_iterations=1000
        self.iteration_size=1000
        self.max_ep_len=200
        self.gamma=0.9
        # model parameters
        self.n_layers=2
        self.layer_size=16
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

class config_frozenlake:
    def __init__(self):
        # environment specific config
        self.d2v = True
        self.rendering = True
        self.use_optimal_baseline = False
        # Timescale parameters
        self.lr_timescale = 1.0
        self.step_timescale = 4
        self.lr_actor = 0.02
        # Training parameters
        self.number_of_iterations=2000
        self.iteration_size=1000
        self.max_ep_len=100
        self.gamma=0.9
        # model parameters
        self.n_layers=0
        self.layer_size=16
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

def get_config(env_name):
    if env_name == 'CartPole-v1':
        return config_cartpole()
    elif env_name == 'FrozenLake-v0':
        return config_frozenlake()
