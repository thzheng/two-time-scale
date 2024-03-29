import os
import sys
import json
import numpy as np
import tensorflow as tf
import gym
from lake_envs import *
import time
import argparse
from model import build_cnn, build_small_cnn, build_configurable_cnn
from model import build_mlp
from config import get_config
from atari_wrappers import wrap_deepmind
from minatar import Make


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', help="gym environment name",
                    default="FrozenLake-v0")
parser.add_argument('-s', '--seed',  default=0, help="randome seed")

def get_result_dir(root_dir):
    dirs = [os.path.basename(x[0]) for x in os.walk(root_dir)]
    if root_dir in dirs:
      dirs.remove(os.path.basename(root_dir))
    nums = []
    for d in dirs:
        try:
            n = int(d)
            nums.append(n)
        except ValueError:
            continue
    if len(nums) == 0:
        return "0"
    dir = str(max(nums) + 1)
    return dir


class MyModel(object):
  def __init__(self, env, config, env_name):
    self.env = env.unwrapped
    self.config=config
    self.d2v = self.config.d2v
    self.use_optimal_baseline = self.config.use_optimal_baseline
    self.rendering = self.config.rendering

    wrap = self.config.wrap
    self.use_minatar = hasattr(self.env.unwrapped, "nchannels")
    if self.use_minatar:
      print("MinAtar doesn't support wrap")
      wrap = False

    # use wrappers
    if wrap:
      self.env = wrap_deepmind(self.env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)

    # use_state_shapeTrue enabled when 3-d (2-d plus RGB) state space used
    self.use_state_shape=False
    if isinstance(self.env.observation_space, gym.spaces.Discrete):
        if self.d2v:
          self.observation_dim = self.env.nS
        else:
          self.observation_dim = 1
    elif len(self.env.observation_space.shape)==1:
      self.observation_dim = self.env.observation_space.shape[0]
    else:
      self.observation_dim = None
      self.use_state_shape = True
      self.state_shape = list(self.env.observation_space.shape)
      assert(len(self.state_shape)==3)

    self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if self.use_minatar:
      print("action_dim discrete")
      self.discrete = True
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
    # Timescale parameters
    self.lr_timescale = self.config.lr_timescale
    self.step_timescale = self.config.step_timescale
    self.lr_actor = self.config.lr_actor
    self.lr_critic = self.lr_actor * self.lr_timescale
    # Training parameters
    self.number_of_iterations=self.config.number_of_iterations
    self.iteration_size=self.config.iteration_size
    self.max_ep_len=self.config.max_ep_len
    self.gamma=self.config.gamma
    # model parameters
    self.use_cnn=self.config.use_cnn
    self.use_small_cnn=self.config.use_small_cnn
    self.n_layers=self.config.n_layers
    self.layer_size=self.config.layer_size

    dir_root = os.path.join("results/", env_name)
    dir_name = get_result_dir(dir_root)
    self.output_path= os.path.join(dir_root, dir_name)
    os.mkdir(self.output_path)
    with open(os.path.join(self.output_path, "config.json"), "w") as f:
        json.dump(vars(self.config), f)
    # Enable multi actors

    # Multi actor params
    self.num_actors = self.config.num_actors
    self.heterogeneity = self.config.heterogeneity
    if self.heterogeneity:
      self.mlp_big_little_config = self.config.mlp_big_little_config
      self.mlp_big_little_map = self.config.mlp_big_little_map
    self.heterogeneity_cnn = self.config.heterogeneity_cnn
    if self.heterogeneity_cnn:
      self.cnn_big_little_config = self.config.cnn_big_little_config
      self.cnn_big_little_map = self.config.cnn_big_little_map
    self.update_actor_ops = []
    self.sampled_action_list = []
    self.actor_loss_list = []
    self.policy_entropy_list = []
    self.lr_actor_op = None
    self.reset_actor_ops = []
    self.perf_ratio = 0.9
    self.reset_interval = self.config.reset_interval

    # Dropout
    self.dropout = self.config.dropout

    assert not(self.use_cnn and self.use_small_cnn)
    # build model
    self.build()

  def add_placeholders_op(self):
    if self.use_state_shape:
      self.observation_placeholder = tf.placeholder(tf.float32, shape=[None, self.state_shape[0], self.state_shape[1], self.state_shape[2]])
    else:
      # observation_placeholder -> N x M, N batches, M env.nS
      self.observation_placeholder = tf.placeholder(tf.float32, shape=[None, self.observation_dim])
    self.action_placeholder = tf.placeholder(tf.int32, shape=[None,])
    self.advantage_placeholder = tf.placeholder(tf.float32, shape=[None,])

  def add_actor_network_op(self, idx, scope = "actor"):
    state_tensor=self.observation_placeholder
    if self.heterogeneity_cnn:
      print(self.cnn_big_little_config[self.cnn_big_little_map[idx]][0])
      state_tensor=build_configurable_cnn(state_tensor, self.cnn_big_little_config[self.cnn_big_little_map[idx]][0], self.cnn_big_little_config[self.cnn_big_little_map[idx]][1], scope)
    elif self.use_cnn:
      state_tensor=build_cnn(state_tensor, scope)
    elif self.use_small_cnn:
      state_tensor=build_small_cnn(state_tensor, scope)

    print("state_tensor.get_shape()")
    print(state_tensor.get_shape())
    if self.heterogeneity:
      action_logits = build_mlp(state_tensor, self.action_dim, scope, self.mlp_big_little_config[self.mlp_big_little_map[idx]][0], self.mlp_big_little_config[self.mlp_big_little_map[idx]][1])
    else:
      dropout=self.dropout
      print("Dropout rate", dropout)
      action_logits = build_mlp(state_tensor, self.action_dim, scope, self.n_layers, self.layer_size, dropout=dropout)
    print("action_logits.get_shape()")
    print(action_logits.get_shape())
    policy_entropy = -tf.reduce_sum(tf.nn.softmax(action_logits) * tf.nn.log_softmax(action_logits), -1)
    print(policy_entropy.get_shape())
    policy_entropy = tf.reduce_sum(policy_entropy)
    #self.sampled_action = tf.squeeze(tf.multinomial(action_logits, 1), axis=1)
    # Add one more sampled action
    self.sampled_action_list.append(tf.squeeze(tf.multinomial(action_logits, 1), axis=1))

    logprob = -1*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_placeholder, logits=action_logits)
    actor_loss = -tf.reduce_mean(logprob * self.advantage_placeholder)
    actor_loss = actor_loss - policy_entropy * 0.0001
    global_step = tf.train.get_or_create_global_step()
    if idx == 0:
      tf.summary.scalar("debug/global_step", global_step)
    print("[actor]num_env_frames", global_step)
    if idx == 0:
      # only the first actor updates lr
      learning_rate = tf.train.polynomial_decay(self.lr_actor, global_step,
                                                  self.config.number_of_iterations, 0)
      self.lr_actor_op = learning_rate
    else:
      learning_rate = self.lr_actor_op
    #learning_rate = tf.train.exponential_decay(self.config.lr_actor,
    #                                           self.config.number_of_iterations,
    #                                           1000, 0.96, staircase=False)
    if idx == 0:
      tf.summary.scalar("lr/actor", learning_rate)
    #self.update_actor_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.actor_loss)
    #self.update_actor_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.actor_loss)
    #self.update_actor_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0, epsilon=0.01).minimize(self.actor_loss, global_step = global_step)

    if idx != 0:
      global_step=None
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0, epsilon=0.01)
    grad_var_pairs = optimizer.compute_gradients(actor_loss)
    vars = [x[1] for x in grad_var_pairs]
    grads = [x[0] for x in grad_var_pairs]
    print("actor", idx)
    print(vars)
    print(grads)
    self.record_grads(idx, vars, grads, "")
    clipped, _ = tf.clip_by_global_norm(grads, 40)
    self.record_grads(idx, vars, clipped, "_clipped")
    #tf.summary.histogram("grad/actor_clipped_{}".format(idx), clipped)
    train_op = optimizer.apply_gradients(zip(clipped, vars), global_step=global_step)

    self.update_actor_ops.append(train_op)

    # add variables for summary
    self.actor_loss_list.append(actor_loss)
    self.policy_entropy_list.append(policy_entropy)

    # add reset op
    self.reset_actor_ops.append(tf.initializers.variables(tf.trainable_variables(scope=scope)))

  def record_grads(self, idx, vars, grads, tag):
      for i, v in enumerate(vars):
        if grads[i] is None:
          # Skip None gradients from other actors
          continue
        tf.summary.histogram("grad{}/actor_{}_{}".format(tag, idx, v), grads[i])

  def add_critic_network_op(self, scope = "critic"):
    state_tensor=self.observation_placeholder
    if self.use_cnn:
      state_tensor=build_cnn(state_tensor, scope)
    elif self.use_small_cnn:
      state_tensor=build_small_cnn(state_tensor, scope)
    self.baseline = tf.squeeze(build_mlp(state_tensor, 1, scope, self.n_layers, self.layer_size), axis=1)
    self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=[None])
    global_step = tf.train.get_or_create_global_step()
    print("[critic]num_env_frames", global_step)
    learning_rate = tf.train.polynomial_decay(self.lr_critic, global_step,
                                                  self.config.number_of_iterations, 0)
    #learning_rate = tf.train.exponential_decay(self.lr_critic,
    #                                           self.config.number_of_iterations,
    #                                           1000, 0.96, staircase=False)
    tf.summary.scalar("lr/critic", learning_rate)
    self.critic_loss = tf.losses.mean_squared_error(self.baseline, self.baseline_target_placeholder, scope=scope)
    #tf.summary.scalar("loss/actor", critic_loss)
    #self.update_critic_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.critic_loss)
    self.update_critic_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0, epsilon=0.01).minimize(self.critic_loss)

  def calculate_advantage(self, returns, observations):
    adv = returns
    #print(returns)
    #print(returns.shape)
    baseline=self.sess.run(self.baseline, feed_dict={self.observation_placeholder: observations})
    baseline_old = baseline

    #print("old baseline", baseline)
    # Use optial baseline
    if self.use_optimal_baseline:
      if len(observations[0]) == 4:
        optimal = [0.063, 0.056, 0.071, 0.052, 0.086, 0.   , 0.11 , 0.   , 0.141, 0.244, 0.297, 0.   , 0.   , 0.378, 0.638, 0.]
      else:
        optimal = [0.254,0.282,0.314,0.349,0.387,0.43,0.478,0.531,0.282,0.314,
0.349,0.387,0.43,0.478,0.531,0.59,0.314,0.349,0.387,0.,0.478,0.531,0.59,0.656,
0.349,0.387,0.43,0.478,0.531,0.,0.656,0.729,0.314,0.349,0.387,0.,
0.59,0.656,0.729,0.81,0.282,0.,0.,0.59,0.656,0.729,0.,0.9,
0.314,0.,0.478,0.531,0.,0.81,0.,1.,0.349,0.387,0.43,0.,0.81,0.9,1.,0.,]
      # TODO(jiale) based on my intuation, this should encourage exploration
      #optimal.reverse()
      #print("observations", observations)
      baseline = np.sum(observations * optimal, axis=1)

    #print("new baseline", baseline)
    #print("old adv", adv - baseline_old)
    adv-=baseline
    #adv = adv - np.mean(adv)
    #print("adv", adv)
    return adv

  def update_critic(self, returns, observations):
    self.sess.run(self.update_critic_op, feed_dict={self.observation_placeholder: observations, self.baseline_target_placeholder: returns})

  def check_critic(self):
    if self.observation_dim is None:
      return
    if self.observation_dim != 1 and not self.d2v:
      return
    #self.env.nS
    if self.d2v:
      l = int(np.sqrt(self.env.nS))
      d = np.eye(self.env.nS)
      np.set_printoptions(linewidth=150, suppress=True)
      values = self.sess.run(self.baseline, feed_dict={self.observation_placeholder: d})
      values = values.reshape((l, l))
      print(values)
      return

    for s in range(self.env.nS):
      if self.d2v:
        #state_vector = [s*4, s*4+1, s*4+2, s*4+3]
        #state_vector = np.eye(self.env.nS)[state_vector]
        state_vector = d
      else:
        state_vector = [[s*4], [s*4+1], [s*4+2], [s*4+3]]

      print(self.sess.run(self.baseline, feed_dict={self.observation_placeholder: state_vector}))

  def update_actor(self, observations, actions, advantages, actor_idx):
    self.sess.run(
      self.update_actor_ops[actor_idx],
      feed_dict={
        self.observation_placeholder: observations,
        self.action_placeholder: actions,
        self.advantage_placeholder : advantages
      })

  def reset_actor(self, actor_idx):
    print("Reset actor: " + str(actor_idx))
    self.sess.run(self.reset_actor_ops[actor_idx])

  def build(self):
    """
    Build the model by adding all necessary variables.
    """
    tf.get_variable(
	'num_environment_frames',
	initializer=tf.zeros_initializer(),
	shape=[],
	dtype=tf.int64,
	trainable=False,
	collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    # add placeholders
    self.add_placeholders_op()
    # create actor
    for i in range(self.num_actors):
      self.add_actor_network_op(i, scope="actor{}".format(i))
    # create critic
    self.add_critic_network_op()

  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables
    """
    # create tf session
    self.sess = tf.Session()
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)

  def add_summary(self):
    """
    Tensorboard stuff.
    """
    # extra placeholders to log stuff from python
    # self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.avg_reward_placeholder_list = []
    for i in range(self.num_actors):
      self.avg_reward_placeholder_list.append(tf.placeholder(tf.float32, shape=(), name="avg_reward/actor_{}".format(i)))
      tf.summary.scalar("reward/Avg_{}".format(i), self.avg_reward_placeholder_list[i])

    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

    # extra summaries from python -> placeholders
    # tf.summary.scalar("reward/Avg", self.avg_reward_placeholder)
    tf.summary.scalar("reward/Max", self.max_reward_placeholder)
    tf.summary.scalar("reward/Std", self.std_reward_placeholder)
    tf.summary.scalar("reward/Eval", self.eval_reward_placeholder)

    for i in range(self.num_actors):
      tf.summary.scalar("debug1/policy_entropy_{}".format(i), self.policy_entropy_list[i])
      tf.summary.scalar("debug/actor_loss_{}".format(i), self.actor_loss_list[i])
    tf.summary.scalar("debug/critic_loss", self.critic_loss)
    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.output_path, self.sess.graph)

  def init_averages(self):
    """
    Defines extra attributes for tensorboard.
    """
    # self.avg_reward = 0.
    self.avg_reward_list = []
    for i in range(self.num_actors):
      self.avg_reward_list.append(0.)
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.

  def update_averages(self, rewards, scores_eval, idx):
    """
    Update the averages.
    """
    #print(rewards)
    # self.avg_reward = np.mean(rewards)
    self.avg_reward_list[idx] = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]

  def record_summary(self, t, observations, actions, advantages, returns):
    """
    Add summary to tensorboard
    """

    fd = {
      # self.avg_reward_placeholder: self.avg_reward,
      self.max_reward_placeholder: self.max_reward,
      self.std_reward_placeholder: self.std_reward,
      self.eval_reward_placeholder: self.eval_reward,
      self.observation_placeholder: observations,
      self.action_placeholder: actions,
      self.advantage_placeholder: advantages,
      self.baseline_target_placeholder: returns,
    }
    for i in range(self.num_actors):
      fd[self.avg_reward_placeholder_list[i]] = self.avg_reward_list[i]
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

  def sample_path(self, env, actor_idx, num_episodes = None, render=False, num_step=None):
    """
    Sample paths (trajectories) from the environment.

    Args:
        num_episodes: the number of episodes to be sampled
            if none, sample one batch
        env: open AI Gym envinronment

    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"
    """
    episode = 0
    episode_rewards = []
    paths = []
    t = 0

    while (num_episodes or t < self.iteration_size):
      state = env.reset()
      states, actions, rewards = [], [], []
      episode_reward = 0

      for step in range(self.max_ep_len):
        if self.d2v:
          state=np.eye(env.nS)[state]
        else:
          if isinstance(self.env.observation_space, gym.spaces.Discrete):
            state = [state]
        states.append(state)
        action = self.sess.run(self.sampled_action_list[actor_idx],
                               feed_dict={self.observation_placeholder : [states[-1]]})[0]
        #env.render()
        state, reward, done, info = env.step(action)
        if render:
            render_state = state[:,:,0]
            render_state = render_state * env.nchannels + env.nchannels/2.0
            #print()
            rs2 = [[str(int(x)) if x != 0 else " " for x in l] for l in render_state]
            frame = ''
            for row in rs2:
                frame = frame + ' '.join(row) + "\n"
            outfile = os.path.join(self.output_path, "{}_{}.txt").format(num_step, actor_idx)
            with open(outfile, "a") as f:
                f.write(frame)
                if done:
                    f.write("-------------------------------------------------\n")
                f.flush()
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.max_ep_len-1):
          episode_rewards.append(episode_reward)
          break
        if (not num_episodes) and t == self.iteration_size:
          break

      path = {"observation" : np.array(states),
                      "reward" : np.array(rewards),
                      "action" : np.array(actions)}
      #print("pathlen", len(states))
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    if render:
      print("wrote rendered txt to {}".format(outfile))
    #print(paths)
    return paths, episode_rewards

  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep

    Args:
            paths: recorded sample paths.  See sample_path() for details.

    Return:
            returns: return G_t for each timestep

    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):

       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

    where T is the last timestep of the episode.

    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]
      returns = []
      curr_reward=0
      for t in range(len(rewards)-1, -1, -1):
        curr_reward*=self.gamma
        curr_reward+=rewards[t]
        returns.append(curr_reward)
      returns.reverse()
      all_returns.append(returns)
    returns = np.concatenate(all_returns)

    return returns


  def train(self):
    """
    Performs training

    """
    last_eval = 0
    last_record = 0
    scores_eval = []

    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time

    # moving avearge of average rewards: p(n+1)=p(n)*self.perf_ratio+reward*(1-self.perf_ratio)
    actor_perf = [0 for i in range(self.num_actors)]

    for t in range(self.number_of_iterations):
      # which actor to reset, if necessary
      reset_actor_idx = None

      for i in range(self.num_actors):
        # collect a batch of samples
        if t % 50 == 1:
            render = True if self.use_minatar else False
        else:
            render = False
        #print("render", render)
        #print("self.use_minatar", self.use_minatar)
        paths, total_rewards = self.sample_path(self.env, i, render=render, num_step=t)
        scores_eval = scores_eval + total_rewards
        observations = np.concatenate([path["observation"] for path in paths])
        #print("observations", observations)
        #print("observations shape", observations.shape)
        actions = np.concatenate([path["action"] for path in paths])
        rewards = np.concatenate([path["reward"] for path in paths])
        # compute Q-val estimates (discounted future returns) for each time step
        returns = self.get_returns(paths)
        #print("returns", returns)
        advantages = self.calculate_advantage(returns, observations)
        #print("advantages", advantages)


        # run training operations
        for step_i in range(self.step_timescale):
          self.update_critic(returns, observations)

        self.update_actor(observations, actions, advantages, i)

        # summary
        self.update_averages(total_rewards, scores_eval, i)
        self.record_summary(t, observations, actions, advantages, returns)

        # compute reward statistics for this batch and log
        avg_reward = np.mean(total_rewards)
        sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
        msg = "[{}] ".format(i) + str(t) + " Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        print(msg)

        # compute moving average
        actor_perf[i]=actor_perf[i]*self.perf_ratio+avg_reward*(1.-self.perf_ratio)
        if reset_actor_idx == None or actor_perf[i] < actor_perf[reset_actor_idx]:
          reset_actor_idx = i

      if (t+1)%10==1:
         self.check_critic()

      if self.reset_interval>0 and  t%self.reset_interval == self.reset_interval - 1:
        self.reset_actor(reset_actor_idx)

    print("- Training done.")

  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training
    episodes.
    """
    if env==None: env = self.env
    paths, rewards = self.sample_path(env, num_episodes, render=True)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    print(msg)
    return avg_reward

  def render_single(self, max_steps=100):
    episode_reward = 0
    ob = self.env.reset()
    for t in range(max_steps):
      self.env.render()
      print("State: ", ob)
      time.sleep(0.25)
      if self.d2v:
        ob=[ob]
        ob=np.eye(self.env.nS)[ob]
      else:
        ob=[[ob]]
      a = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder: ob})[0]
      ob, rew, done, _ = self.env.step(a)
      episode_reward += rew
      if done:
        break
    self.env.render();
    if not done:
      print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
      print("Episode reward: %f" % episode_reward)


  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()
    # model
    self.train()
    # evaluate
    # self.evaluate()
    if self.rendering:
      self.render_single()

if __name__ == '__main__':
    args = parser.parse_args()
    seed = int(args.seed)
    tf.random.set_random_seed(seed)

    if "-v" in args.env:
      env = gym.make(args.env)
      env.seed(seed)
    else:
      env = Make(args.env, seed)

    config = get_config(args.env)
    model = MyModel(env, config, args.env)
    model.run()
