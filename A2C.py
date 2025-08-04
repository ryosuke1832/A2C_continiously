import gym
# import gym_soccer  # Unity環境を使うためコメントアウト
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import time
from collections import deque


## Code is from: https://github.com/shakedzy/notebooks/blob/master/gradient_policy_and_actor_critic/Acrobot%20with%20Actor-Critic.ipynb
## Blog: https://towardsdatascience.com/qrash-course-ii-from-q-learning-to-gradient-policy-actor-critic-in-12-minutes-8e8b47129c8c

tf.disable_eager_execution()
tf.disable_v2_behavior()

# Streams
# Init LSL streams
# streams_robot = resolve_stream('name','RobotData')
# print("Crete Stream 1")
# stream_sigma= StreamInfo('MarkerStream', 'Sigma', 1, 0, 'string', 'xxx')
# print("Crete Stream 1")
# streams_PEN = resolve_stream('type', 'Markers', 'name','PEN')
# print("Crete Stream 1")


# Setting up environment
# env = gym.make('Acrobot-v11')
# env = gym.make('MountainCar-v0')
# env=gym.make('pHRC-v0')
# env=gym.make('Soccer-v0')
# env.env.init(streams_robot,streams_PEN,stream_sigma)
# state_size =4 # env.observation_space.shape[0]
# num_of_actions =2 # env.action_space.n


# Unity-ROS環境用の新しい設定
class UnityGraspEnvironment:
    def __init__(self):
        # LSLストリーム設定
        self.grip_action_stream = StreamInfo('GripAction', 'Force', 1, 0, 'float32', 'grip_control')
        self.grip_outlet = StreamOutlet(self.grip_action_stream)
        
        # Unityからの入力ストリーム
        self.reward_inlet = None
        self.state_inlet = None
        self.setup_unity_streams()
        
        # 環境パラメータ
        self.state_size = 6
        self.action_space_low = 0.0
        self.action_space_high = 25.0
        self.current_state = np.zeros(self.state_size)
        self.episode_start_time = 0
        self.max_episode_time = 30.0
    
    def setup_unity_streams(self):
        try:
            reward_streams = resolve_stream('name', 'GraspReward')
            if reward_streams:
                self.reward_inlet = StreamInlet(reward_streams[0])
                print("報酬ストリーム接続完了")
            
            state_streams = resolve_stream('name', 'GraspState')
            if state_streams:
                self.state_inlet = StreamInlet(state_streams[0])
                print("状態ストリーム接続完了")
        except Exception as e:
            print(f"Unity ストリーム接続エラー: {e}")
    
    def reset(self):
        # リセット信号をUnityに送信
        self.grip_outlet.push_sample([0.0])
        time.sleep(0.5)
        
        # 初期状態を取得
        if self.state_inlet:
            sample, _ = self.state_inlet.pull_sample(timeout=1.0)
            if sample:
                self.current_state = np.array(sample)
            else:
                self.current_state = np.zeros(self.state_size)
        
        self.episode_start_time = time.time()
        return self.current_state
    
    def step(self, action):
        # アクションをUnityに送信
        grip_force = np.clip(action, self.action_space_low, self.action_space_high)
        self.grip_outlet.push_sample([float(grip_force)])
        
        # 応答待機
        time.sleep(0.1)
        
        # 新しい状態を取得
        next_state = np.zeros(self.state_size)
        if self.state_inlet:
            sample, _ = self.state_inlet.pull_sample(timeout=0.2)
            if sample:
                next_state = np.array(sample)
        
        # 報酬を取得
        reward = 0.0
        if self.reward_inlet:
            sample, _ = self.reward_inlet.pull_sample(timeout=0.2)
            if sample:
                reward = self.convert_unity_reward(sample[0])
        
        # エピソード終了判定
        elapsed_time = time.time() - self.episode_start_time
        done = elapsed_time > self.max_episode_time or abs(reward) > 50
        
        self.current_state = next_state
        return next_state, reward, done, {}
    
    def convert_unity_reward(self, unity_result):
        if unity_result == 0:    # Success
            return 100.0
        elif unity_result == 1:  # Over-grip
            return -50.0
        elif unity_result == 2:  # Under-grip
            return -50.0
        elif unity_result == 3:  # Failure
            return -100.0
        else:
            return -1.0  # ステップペナルティ

# Unity環境のインスタンス化
env = UnityGraspEnvironment()
state_size = env.state_size
num_of_actions = 1  # 連続制御（1次元）


# Setting Game reward
end_game_reward = 100

# ===== 4. QNetwork クラスの修正 =====  
class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, num_of_actions=1):
        # 連続制御用に修正
        self.actions = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='dqn_actions')
        self.future_actions = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='dqn_future_actions')
        self.q_target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='dqn_q_target')
        self.r = tf.placeholder(shape=None, dtype=tf.float32, name='dqn_r')
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='dqn_states')
        
        # ネットワーク構築
        _layer = self.states
        for i in range(len(hidden_layers_size)):
            l = hidden_layers_size[i]
            _layer = tf.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                   kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                   name='dqn_layer_{}'.format(i))
        
        self.last_layer = tf.layers.dense(inputs=_layer, units=1, activation=None,
                                        kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                        name='dqn_output')
        
        # 損失関数
        self.predictions = self.last_layer  # 連続値出力
        self.labels = self.r + gamma * self.q_target
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


# ===== 3. PolicyGradient クラスの修正 =====
class PolicyGradient:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, num_of_actions=None, min_value=0, max_value=25):
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='pg_states')
        self.q = tf.placeholder(shape=None, dtype=tf.float32, name='pg_q')
        self.actions = tf.placeholder(shape=None, dtype=tf.float32, name='pg_actions')
        
        # ネットワーク構築
        _layer = self.states
        for i in range(len(hidden_layers_size)):
            l = hidden_layers_size[i]
            _layer = tf.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                   kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                   name='pg_layer_{}'.format(i))
        
        # 連続制御用の出力層
        self.raw_predictions = tf.layers.dense(inputs=_layer, units=1, activation=tf.nn.sigmoid,
                                             kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                                             name='pg_output')
        
        # アクション範囲にスケール
        self.predictions = min_value + self.raw_predictions * (max_value - min_value)
        
        # 損失関数（連続制御用）
        self.log_policy = tf.log(tf.clip_by_value(self.raw_predictions, 1e-10, 1.0))
        self.cost = -tf.reduce_mean(self.q * self.log_policy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

# ===== 5. ActorCritic クラスの修正 =====
class ActorCritic:
    def __init__(self, input_size, num_of_actions, actor_hidden_layers_size, critic_hidden_layers_size,
                 actor_gamma, critic_gamma, actor_learning_rate, critic_learning_rate):
        
        # 連続制御用に修正
        self.critic = QNetwork(critic_hidden_layers_size, critic_gamma, critic_learning_rate, input_size, 1)
        self.actor = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size, 
                                  min_value=env.action_space_low, max_value=env.action_space_high)
        self.memory = []
        self.num_of_actions = num_of_actions

    def _extract_from_batch(self, batch, key):
        return np.array(list(map(lambda x: x[key], batch)))

    def _update_terminal_states(self, q_target, terminals):
        for i in range(len(terminals)):
            if terminals[i]:
                q_target[i] = 0.0
        return q_target

    def remember(self, **kwargs):
        self.memory.append(kwargs)

    def act(self, session, state, greedy=False):
        # 連続制御用のアクション選択
        action_value = session.run(self.actor.predictions,
                                 feed_dict={self.actor.states: np.expand_dims(state, axis=0)}).flatten()
        
        if not greedy:
            # 探索ノイズを追加
            noise = np.random.normal(0, 0.1 * (env.action_space_high - env.action_space_low))
            action_value += noise
            action_value = np.clip(action_value, env.action_space_low, env.action_space_high)
        
        return action_value[0]

    def learn(self, session):
        batch = shuffle(self.memory)
        next_states = self._extract_from_batch(batch, 'next_state')
        states = self._extract_from_batch(batch, 'state')
        rewards = self._extract_from_batch(batch, 'reward')
        actions = self._extract_from_batch(batch, 'action')
        terminals = self._extract_from_batch(batch, 'game_over')

        # Q値計算
        q = session.run(self.critic.last_layer, feed_dict={self.critic.states: states})
        q_t = session.run(self.critic.last_layer, feed_dict={self.critic.states: next_states})
        q_t = self._update_terminal_states(q_t.flatten(), terminals=terminals)

        # Actor更新用のアクション確率
        next_action_probs = session.run(self.actor.predictions, 
                                       feed_dict={self.actor.states: next_states})

        # 学習実行
        actor_cost, _, critic_cost, _ = session.run([self.actor.cost, self.actor.optimizer,
                                                   self.critic.cost, self.critic.optimizer],
                                                  feed_dict={
                                                      self.actor.states: states,
                                                      self.actor.q: q.flatten(),
                                                      self.actor.actions: actions,
                                                      self.critic.q_target: q_t.reshape(-1, 1),
                                                      self.critic.r: rewards,
                                                      self.critic.actions: actions.reshape(-1, 1),
                                                      self.critic.future_actions: next_action_probs,
                                                      self.critic.states: states})

        if np.isnan(actor_cost) or np.isnan(critic_cost):
            raise Exception('NaN cost!')

        self.memory = []
        return actor_cost, critic_cost




# ===== 6. ハイパーパラメータの修正 =====
# Unity把持タスク用のパラメータ
actor_hidden_layers = [64, 32]
critic_hidden_layers = [64, 64, 32]
learning_rate = 0.001
gamma = 0.99


# ===== 7. エージェント初期化の修正 =====
sess = tf.Session()
ac = ActorCritic(input_size=state_size, num_of_actions=num_of_actions,
                actor_hidden_layers_size=actor_hidden_layers, 
                critic_hidden_layers_size=critic_hidden_layers,
                actor_gamma=gamma, critic_gamma=gamma,
                actor_learning_rate=learning_rate, critic_learning_rate=learning_rate)

game_df = pd.DataFrame(columns=['game','steps','actor_cost','critic_cost','final_reward','action'])
sess.run(tf.global_variables_initializer())

# ===== 8. 学習ループの修正 =====
def print_stuff(s, every=10):
    if game % every == 0 or game == 1:
        print(s)

# Unity用学習パラメータ
games = 100
max_steps_per_episode = 200

print("Unity-ROS A2C学習開始...")
print("Unity側でエピソードを開始してください")

for g in range(games):
    game = g + 1
    game_over = False
    
    # Unity環境リセット
    next_state = env.reset()
    print(f"Episode {game}: Unity環境リセット完了")
    
    steps = 0
    total_reward = 0
    episode_actions = []
    
    while not game_over and steps < max_steps_per_episode:
        steps += 1
        state = np.copy(next_state)
        
        # アクション選択
        action = ac.act(sess, state)
        episode_actions.append(action)
        
        # 環境でステップ実行
        next_state, r, game_over, _ = env.step(action)
        total_reward += r
        
        # 経験保存
        ac.remember(state=state, action=action, reward=r, 
                   next_state=next_state, game_over=game_over)
        
        # 学習実行
        if len(ac.memory) > 0:
            actor_cost, critic_cost = ac.learn(sess)
        else:
            actor_cost, critic_cost = 0, 0
        
        # 進行状況表示
        if steps % 10 == 0:
            print(f"  Step {steps}: Action={action:.2f}N, Reward={r:.1f}, Total={total_reward:.1f}")
        
        # Unity側が終了を要求した場合
        if game_over:
            print(f"  Unity側がエピソード終了を要求")
            break
    
    # エピソード結果記録
    avg_action = np.mean(episode_actions) if episode_actions else 0
    print_stuff(f'Game {game} 完了: {steps}ステップ, 総報酬: {total_reward:.1f}, '
               f'平均アクション: {avg_action:.2f}N, Actor損失: {actor_cost:.2e}, Critic損失: {critic_cost:.2e}')
    
    game_df = game_df.append({
        'game': game, 
        'steps': steps, 
        'actor_cost': actor_cost, 
        'critic_cost': critic_cost,
        'final_reward': total_reward,
        'action': avg_action
    }, ignore_index=True)
    
    # 定期的な結果保存
    if game % 20 == 0:
        game_df.to_csv(f'unity_a2c_progress_game{game}.csv', index=False)
        print(f"進行状況をファイル保存: unity_a2c_progress_game{game}.csv")

# ===== 9. 結果保存と可視化 =====
# 最終結果保存
game_df.to_csv('unity_a2c_final_results.csv', index=False)
print("最終結果保存完了: unity_a2c_final_results.csv")

# 学習曲線の可視化
if len(game_df) > 0:
    # ステップ数の移動平均
    window_size = min(10, len(game_df))
    game_df['steps_moving_average'] = game_df['steps'].rolling(window=window_size).mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ステップ数の推移
    axes[0,0].plot(game_df['game'], game_df['steps_moving_average'])
    axes[0,0].set_title('Steps per Episode (Moving Average)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Steps')
    
    # 総報酬の推移
    game_df['reward_moving_average'] = game_df['final_reward'].rolling(window=window_size).mean()
    axes[0,1].plot(game_df['game'], game_df['reward_moving_average'])
    axes[0,1].set_title('Total Reward per Episode (Moving Average)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Total Reward')
    
    # Actor損失の推移
    game_df['actor_cost_moving_average'] = game_df['actor_cost'].rolling(window=window_size).mean()
    axes[1,0].plot(game_df['game'], game_df['actor_cost_moving_average'])
    axes[1,0].set_title('Actor Cost (Moving Average)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Actor Cost')
    
    # 平均アクションの推移
    game_df['action_moving_average'] = game_df['action'].rolling(window=window_size).mean()
    axes[1,1].plot(game_df['game'], game_df['action_moving_average'])
    axes[1,1].set_title('Average Action per Episode (Moving Average)')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Average Grip Force (N)')
    
    plt.tight_layout()
    plt.savefig('unity_a2c_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===== 10. 学習済みモデルでのテスト =====
print("\n学習済みモデルでテスト実行...")
test_episodes = 5

for test_ep in range(test_episodes):
    print(f"\nテストエピソード {test_ep+1}:")
    next_state = env.reset()
    game_over = False
    steps = 0
    total_reward = 0
    
    while not game_over and steps < 50:
        steps += 1
        state = np.copy(next_state)
        
        # 貪欲行動（探索なし）
        action = ac.act(sess, state, greedy=True)
        next_state, reward, game_over, _ = env.step(action)
        
        total_reward += reward
        
        print(f"  Step {steps}: Action={action:.2f}N, Reward={reward:.1f}")
        
        if abs(reward) > 50:  # 成功/失敗で終了
            break
    
    print(f"テストエピソード {test_ep+1} 結果: 総報酬={total_reward:.1f}, ステップ数={steps}")

print("\nUnity-ROS A2C学習完了!")

# ===== 11. モデル保存機能（オプション） =====
def save_model(session, save_path="unity_a2c_model"):
    """学習済みモデルを保存"""
    saver = tf.train.Saver()
    saver.save(session, save_path)
    print(f"モデル保存完了: {save_path}")

def load_model(session, save_path="unity_a2c_model"):
    """保存されたモデルを読み込み"""
    saver = tf.train.Saver()
    saver.restore(session, save_path)
    print(f"モデル読み込み完了: {save_path}")

# 学習後にモデルを保存
save_model(sess, "unity_grasp_a2c_model")

print("\n使用方法:")
print("1. Unity側でUnityA2CManager.csが設定されたシーンを開く")
print("2. Unity側でPlay開始")
print("3. このPythonスクリプトを実行")
print("4. Unity側で'Start Episode Manually'を実行するか、自動開始を待つ")
print("5. 学習進行をコンソールで監視")