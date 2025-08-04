
import gym
import gym_soccer
import random
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import matplotlib.pyplot as plt
from collections import deque
import time
import scipy.io as sio

tf.disable_eager_execution()
tf.disable_v2_behavior()
#env = gym.make('MountainCarContinuous-v0')

env=gym.make('Soccer-v0')
# env.env.init(streams_robot,streams_PEN,stream_sigma)
#state_size =4 # env.observation_space.shape[0]
#num_of_actions =2 # env.action_space.n

state_size = env.observation_space.shape[0]


# The CRITIC
class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, name, reg_lambda):
        self.q_target = tf.placeholder(shape=None, dtype=tf.float32, name='dqn_%s_q_target' % name)
        self.r = tf.placeholder(shape=None, dtype=tf.float32, name='dqn_%s_r' % name)
        self.actions = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='dqn_%s_actions' % name)
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='dqn_%s_states' % name)

        _layer = tf.concat([self.states, self.actions], axis=1)
        for i in range(len(hidden_layers_size)):
            l = hidden_layers_size[i]
            _layer = tf.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                     kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                     name='dqn_{n}_layer_{i}'.format(n=name, i=i))
        self.predictions = tf.layers.dense(inputs=_layer, units=1, activation=None,  # Linear activation
                                           kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                           name='dqn_%s_last_layer' % name)

        self.labels = self.r + gamma * self.q_target
        self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                            if 'bias' not in v.name and 'dqn_%s' % name in v.name])
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.cost = self.cost + reg_lambda * self.l2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

# THE ACTOR
class PolicyGradient:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, min_value, max_value, name):
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='pg_states')
        self.q = tf.placeholder(shape=None, dtype=tf.float32, name='pg_q')
        self.actions = tf.placeholder(shape=None, dtype=tf.float32, name='pg_actions')

        _layer = self.states
        for i in range(len(hidden_layers_size)):
            l = hidden_layers_size[i]
            _layer = tf.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                     kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                     name='pg_{n}_layer_{i}'.format(n=name, i=i))
        self.predictions = tf.layers.dense(inputs=_layer, units=1, activation=None,
                                           kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                           name='pg_%s_last_layer' % name)

        self.predictions = tf.clip_by_value(self.predictions, min_value, max_value)
        self.cost = tf.reduce_mean(self.q * self.predictions)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


# Experience Replay Memory
class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)
        self.counter = 0

    def __len__(self):
        return len(self.memory)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        if n > len(self.memory): n = len(self.memory)
        return random.sample(self.memory, n)

# Exploration Noise
class OUNoise:
    def __init__(self, mu, theta, sigma):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = self.mu

    def noise(self):  # dt = 1
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.normal()
        self.x += dx
        return abs(self.x)

# ACTOR-CRITIC (DDPG)
class ActorCritic:
    def __init__(self, input_size, memory_size, actor_hidden_layers_size, critic_hidden_layers_size,
                 actor_gamma, critic_gamma, critic_reg_lambda,
                 actor_learning_rate, critic_learning_rate, tau, epochs_to_copy_target, ou_mu, ou_theta, ou_sigma):
        self.critic = QNetwork(critic_hidden_layers_size, critic_gamma, critic_learning_rate, input_size,
                               name='online', reg_lambda=critic_reg_lambda)
        self.actor = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size,
                                    min_value=env.action_space.low, max_value=env.action_space.high,
                                    name='online')
        # self.actor = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size,
        #                             min_value=env.action_space.low, max_value=env.action_space.high,
        #                             name='online')
        self.critic_target = QNetwork(critic_hidden_layers_size, critic_gamma, critic_learning_rate, input_size,
                                      name='target', reg_lambda=critic_reg_lambda)
        self.actor_target = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size,
                                           min_value=env.action_space.low, max_value=env.action_space.high,
                                           name='target')
        # self.actor_target = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size,
        #                                    min_value=env.action_space.low, max_value=env.action_space.high,
        #                                    name='target')
        self.memory = ReplayMemory(memory_size)
        self.tau = tau
        self.ou = OUNoise(mu=ou_mu, theta=ou_theta, sigma=ou_sigma)
        self.epoch_counter = 0
        self.epochs_to_copy_target = epochs_to_copy_target

    def _extract_from_batch(self, batch, key):
        return np.array(list(map(lambda x: x[key], batch)))

    def _update_terminal_states(self, q_target, terminals):
        for i in range(len(terminals)):
            if terminals[i]:
                q_target[i] = 0.0
        return q_target

    def _copy_networks(self, session):
        tf_vars = tf.trainable_variables()
        num_of_vars = len(tf_vars)
        operations = []
        for i, v in enumerate(tf_vars[0:num_of_vars // 2]):
            operations.append(tf_vars[i + num_of_vars // 2].assign(
                (v.value() * self.tau) + ((1 - self.tau) * tf_vars[i + num_of_vars // 2].value())))
        session.run(operations)

    def reset_noise(self):
        self.ou.reset()

    def remember(self, **kwargs):
        self.memory.append(kwargs)

    def act(self, session, state, greedy=False):
        if greedy:
            action = session.run(self.actor.predictions,
                                 feed_dict={self.actor.states: np.expand_dims(state, axis=0)}).flatten()
        else:
            action = session.run(self.actor.predictions,
                                 feed_dict={self.actor.states: np.expand_dims(state, axis=0)}).flatten()
            action += self.ou.noise()
        return action

    def learn(self, session, batch_size):
        self.epoch_counter += 1
        if len(self.memory) >= batch_size:
            batch = self.memory.sample(batch_size)
            next_states = self._extract_from_batch(batch, 'next_state')
            states = self._extract_from_batch(batch, 'state')
            rewards = self._extract_from_batch(batch, 'reward')
            actions = self._extract_from_batch(batch, 'action')
            terminals = self._extract_from_batch(batch, 'game_over')

            next_actions = session.run(self.actor_target.predictions, feed_dict={self.actor_target.states: next_states})
            q_t = session.run(self.critic_target.predictions, feed_dict={self.critic_target.states: next_states,
                                                                         self.critic_target.actions: next_actions})
            q_t = self._update_terminal_states(q_t, terminals=terminals)
            q = session.run(self.critic.predictions, feed_dict={self.critic.states: states,
                                                                self.critic.actions: actions})

            actor_cost, _, critic_cost, _ = session.run([self.actor.cost, self.actor.optimizer,
                                                         self.critic.cost, self.critic.optimizer],
                                                        feed_dict={self.actor.states: states,
                                                                   self.actor.q: q,
                                                                   self.actor.actions: actions,
                                                                   self.critic.q_target: q_t,
                                                                   self.critic.r: rewards,
                                                                   self.critic.actions: actions,
                                                                   self.critic.states: states})
            if np.isnan(actor_cost) or np.isnan(critic_cost): raise Exception('NaN cost!')
            if self.epoch_counter % self.epochs_to_copy_target == 0: self._copy_networks(session)
            return actor_cost, critic_cost
        else:
            return 0, 0

# Hyperparameters
actor_hidden_layers = [36]
critic_hidden_layers = [24]
actor_learning_rate =0.01 # 1e-4
critic_learning_rate =0.1 # 1e-3
critic_reg_lambda = 0.0  # no regularization
gamma = 0.99
batch_size =20 # 100
memory_size = batch_size * 10

# Running-time optimization
epochs_to_copy_target = 1000
tau = 1

# Ornstein–Uhlenbeck noise parameters
ou_mu = 0
ou_theta = 0.15
ou_sigma = 0.2


sess = tf.Session()
ac = ActorCritic(input_size=state_size, memory_size=memory_size,
                 actor_hidden_layers_size=actor_hidden_layers, critic_hidden_layers_size=critic_hidden_layers,
                 actor_gamma=gamma, critic_gamma=gamma, critic_reg_lambda=critic_reg_lambda,
                 actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate,
                 tau=tau, epochs_to_copy_target=epochs_to_copy_target,
                 ou_mu=ou_mu, ou_theta=ou_theta, ou_sigma=ou_sigma)


game_df = pd.DataFrame(columns=['game','steps','actor_cost','critic_cost'])
sess.run(tf.global_variables_initializer())


def print_stuff(s, every=10):
    if game % every == 0 or game == 1:
        print(s)


games = 100
env._setTimes(time.time(), 15)
for g in range(games):
    game = g + 1
    game_over = False
    next_state = env.reset()
    steps = 0
    ac.reset_noise()
    while not game_over:
        steps += 1
        state = np.copy(next_state)
        action = ac.act(sess, state)
        next_state, r, game_over, _ = env.step(action)
        ac.remember(state=state, action=action, reward=r, next_state=next_state, game_over=game_over)
        actor_cost, critic_cost = ac.learn(sess, batch_size)
        print('Action and Critic ' + str(actor_cost)+' '+str(critic_cost))
    print_stuff('Game {g} ended after {s} steps | Actor cost: {a:.2e}, Critic cost: {c:.2e}'.format(g=game, s=steps, a=actor_cost, c=critic_cost))
    game_df = game_df.append({'game':game, 'steps':steps, 'actor_cost':actor_cost, 'critic_cost':critic_cost,'action':action},
                             ignore_index=True)

game_df.to_csv('RL_data.csv',index=False)

game_df['steps_moving_average'] = game_df['steps'].rolling(window=50).mean()
ax = game_df.plot('game','steps_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Steps')
plt.show()




game_df['actor_cost_moving_average'] = game_df['actor_cost'].rolling(window=50).mean()
ax = game_df.plot('game','actor_cost_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Actor Cost')
plt.show()


game_df['critic_cost_moving_average'] = game_df['critic_cost'].rolling(window=50).mean()
ax = game_df.plot('game','critic_cost_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Critic Cost')
plt.show()



next_state = env.reset()
env.render()
game_over = False
steps = 0
ac.reset_noise()
while not game_over:
    steps += 1
    state = np.copy(next_state)
    action = ac.act(sess, state)
    next_state, _, game_over, _ = env.step(action)
    env.render()
print('Ended after {} steps'.format(steps))