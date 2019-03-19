import tensorflow as tf

class config_cartpole:
    def __init__(self):
        # environment specific config
        self.d2v = False
        self.rendering = False
        self.use_optimal_baseline = False
        self.wrap=False
        # Timescale parameters
        self.lr_timescale = 10.0
        self.step_timescale = 1
        self.lr_actor = 0.0001
        # Training parameters
        self.number_of_iterations=1000
        self.iteration_size=1000
        self.max_ep_len=200
        self.gamma=0.9
        # model parameters
        self.use_cnn=False
        self.use_small_cnn=False
        self.n_layers=2
        self.layer_size=16
        self.num_actors=1
        self.heterogeneity=False
        self.heterogeneity_cnn=False
        # 0 never reset
        self.reset_interval=0
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

class config_frozenlake:
    def __init__(self):
        # environment specific config
        # use one hot vectors for state
        self.d2v = True
        self.rendering = True
        self.use_optimal_baseline = False
        self.wrap=False
        # Timescale parameters
        self.lr_timescale = 1.0
        self.step_timescale = 1
        self.lr_actor = 0.0001
        # Training parameters
        self.number_of_iterations=2000
        self.iteration_size=1000
        self.max_ep_len=100
        self.gamma=0.9
        # model parameters
        self.use_cnn=False
        self.use_small_cnn=False
        self.n_layers=1
        self.layer_size=16
        self.num_actors=1
        self.heterogeneity=False
        self.heterogeneity_cnn=False
        # 0 never reset
        self.reset_interval=0
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

class config_pong:
    def __init__(self):
        # environment specific config
        self.d2v = False
        self.rendering = False
        self.use_optimal_baseline = False
        self.wrap=True
        # Timescale parameters
        self.lr_timescale = 1.0
        self.step_timescale = 1
        self.lr_actor = 0.001
        # Training parameters
        self.number_of_iterations=int(1e8)
        self.iteration_size=1000
        self.max_ep_len=-1
        self.gamma=0.99
        # model parameters
        self.use_cnn=True
        self.use_small_cnn=False
        self.n_layers=2
        self.layer_size=64
        self.num_actors=1
        self.heterogeneity=False
        self.heterogeneity_cnn=False
        # 0 never reset
        self.reset_interval=0
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

class config_continuous:
    def __init__(self):
        # environment specific config
        self.d2v = False
        self.rendering = False
        self.use_optimal_baseline = False
        self.wrap=False
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
        self.use_cnn=False
        self.use_small_cnn=False
        self.n_layers=2
        self.layer_size=16
        self.num_actors=1
        self.heterogeneity=False
        self.heterogeneity_cnn=False
        # 0 never reset
        self.reset_interval=0
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

class config_atari:
    def __init__(self):
        # environment specific config
        self.d2v = False
        self.rendering = False
        self.use_optimal_baseline = False
        self.wrap=True
        # Timescale parameters
        self.lr_timescale = 1.0
        self.step_timescale = 1
        self.lr_actor = 0.0006
        # Training parameters
        self.number_of_iterations=int(1e8)
        self.iteration_size=1500
        self.max_ep_len=-1
        self.gamma=0.99
        # model parameters
        self.use_cnn=True
        self.use_small_cnn=False
        self.n_layers=1
        self.layer_size=512
        self.num_actors=1
        self.heterogeneity=False
        self.heterogeneity_cnn=False
        # 0 never reset
        self.reset_interval=0
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

class config_minatar:
    def __init__(self):
        # environment specific config
        # use one hot vectors for state
        self.d2v = False
        self.rendering = False
        self.use_optimal_baseline = False
        self.wrap=False
        # Timescale parameters
        self.lr_timescale = 1.0
        self.step_timescale = 1
        self.lr_actor = 0.01
        # Training parameters
        self.number_of_iterations=int(1e4)
        self.iteration_size=1000
        self.max_ep_len=-1
        self.gamma=0.99
        # model parameters
        self.use_cnn=False
        self.use_small_cnn=True
        self.n_layers=1
        self.layer_size=512
        self.num_actors=4
        self.heterogeneity=True
        # [[# layers, layer size] * # of configs]
        self.mlp_big_little_config=[[2, 256], [1, 128], [1, 64]]
        self.mlp_big_little_map=[0, 0, 0, 0]
        self.heterogeneity_cnn=True
        self.cnn_big_little_config=[[[64, 128], [3, 5]], [[32, 64], [3, 3]]]
        self.cnn_big_little_map=[0, 0, 0, 0]
        # 0 never reset
        self.reset_interval=2500
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.iteration_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.iteration_size

def get_config(env_name):
    if env_name == 'CartPole-v1':
        return config_cartpole()
    elif env_name == 'FrozenLake-v0':
        return config_frozenlake()
    elif env_name == 'Pong-v0':
        return config_pong()
    elif "FrozenLake" in env_name:
        return config_frozenlake()
    elif "Breakout" in env_name:
        return config_atari()
    else:
        print("using config_minatar")
        return config_minatar()
