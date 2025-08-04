# Unity-ROS A2C システム（EEGなし版）
import gym
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow as tf2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
from collections import deque
import random
import time

tf.disable_eager_execution()
tf.disable_v2_behavior()

# Unity-ROS連携のためのストリーム設定
class UnityROSInterface:
    def __init__(self):
        # Unity側の把持システムとの通信
        self.grip_action_stream = StreamInfo('GripAction', 'Force', 1, 0, 'float32', 'grip_control')
        self.grip_outlet = StreamOutlet(self.grip_action_stream)
        
        # Unity側からの報酬ストリーム（把持結果）
        self.reward_inlet = None
        self.state_inlet = None
        
        self.setup_unity_streams()
        
    def setup_unity_streams(self):
        """Unity側からの報酬・状態ストリームを接続"""
        try:
            # Unity側からの報酬ストリーム
            reward_streams = resolve_stream('name', 'GraspReward')
            if reward_streams:
                self.reward_inlet = StreamInlet(reward_streams[0])
                print("把持報酬ストリーム接続完了")
            
            # Unity側からの状態ストリーム 
            state_streams = resolve_stream('name', 'GraspState')
            if state_streams:
                self.state_inlet = StreamInlet(state_streams[0])
                print("把持状態ストリーム接続完了")
                
        except Exception as e:
            print(f"Unity ストリーム接続エラー: {e}")
    
    def send_grip_action(self, force_value):
        """Unity側に把持力指令を送信"""
        self.grip_outlet.push_sample([force_value])
    
    def get_reward_from_unity(self):
        """Unity側から報酬を取得"""
        if self.reward_inlet:
            sample, timestamp = self.reward_inlet.pull_sample(timeout=0.1)
            if sample:
                return self.convert_unity_reward(sample[0])
        return 0.0
    
    def get_state_from_unity(self):
        """Unity側から状態を取得"""
        if self.state_inlet:
            sample, timestamp = self.state_inlet.pull_sample(timeout=0.1)
            if sample:
                return np.array(sample)
        return np.zeros(6)  # デフォルト状態
    
    def convert_unity_reward(self, unity_result):
        """Unity側の把持結果を報酬に変換"""
        # Unity側の把持評価結果に基づく報酬設計
        if unity_result == 0:  # Success
            return 100.0
        elif unity_result == 1:  # Over-grip
            return -50.0
        elif unity_result == 2:  # Under-grip
            return -50.0
        elif unity_result == 3:  # Failure
            return -100.0
        else:
            return 0.0

# カスタムUnity把持環境
class UnityGraspEnvironment:
    def __init__(self, unity_interface):
        self.unity = unity_interface
        # 状態空間：[グリッパー位置, 把持力, 物体接触, 変形度, 力誤差, 時間]
        self.state_size = 6
        # 行動空間：連続的な把持力制御
        self.action_space_low = np.array([0.0])   # 最小把持力
        self.action_space_high = np.array([25.0]) # 最大把持力
        
        self.current_state = np.zeros(self.state_size)
        self.episode_start_time = 0
        self.max_episode_time = 30.0  # 30秒でエピソード終了
        
    def reset(self):
        """エピソードリセット"""
        # Unity側にリセット信号送信
        self.unity.send_grip_action(0.0)
        time.sleep(0.5)  # Unity側の初期化待ち
        
        self.current_state = self.unity.get_state_from_unity()
        self.episode_start_time = time.time()
        
        return self.current_state
    
    def step(self, action):
        """アクション実行とステップ更新"""
        # 把持力を正規化してUnity側に送信
        grip_force = np.clip(action[0], self.action_space_low[0], self.action_space_high[0])
        self.unity.send_grip_action(grip_force)
        
        # Unity側からの応答を待機
        time.sleep(0.1)
        
        # 新しい状態を取得
        next_state = self.unity.get_state_from_unity()
        
        # 報酬を取得
        reward = self.unity.get_reward_from_unity()
        
        # エピソード終了条件
        elapsed_time = time.time() - self.episode_start_time
        done = elapsed_time > self.max_episode_time or abs(reward) > 50
        
        self.current_state = next_state
        
        return next_state, reward, done, {}

# A2C Critic Network
class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, name='', reg_lambda=0.0):
        self.actions = tf.placeholder(shape=(None, 1), dtype=tf.float32, name=f'critic_{name}_actions')
        self.q_target = tf.placeholder(shape=(None, 1), dtype=tf.float32, name=f'critic_{name}_q_target')
        self.r = tf.placeholder(shape=None, dtype=tf.float32, name=f'critic_{name}_r')
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name=f'critic_{name}_states')
        
        # ネットワーク構築
        _layer = self.states
        for i, layer_size in enumerate(hidden_layers_size):
            _layer = tf.layers.dense(
                inputs=_layer, 
                units=layer_size, 
                activation=tf.nn.relu,
                kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                name=f'critic_{name}_layer_{i}'
            )
        
        self.predictions = tf.layers.dense(
            inputs=_layer, 
            units=1, 
            activation=None,
            kernel_initializer=tf2.keras.initializers.GlorotNormal(),
            name=f'critic_{name}_output'
        )
        
        # 損失関数とオプティマイザー
        self.labels = self.r + gamma * self.q_target
        
        # L2正則化
        self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                           if 'bias' not in v.name and f'critic_{name}' in v.name])
        
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=self.labels, predictions=self.predictions))
        self.cost = self.cost + reg_lambda * self.l2
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

# A2C Actor Network（連続制御用）
class PolicyGradient:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, min_value, max_value, name=''):
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name=f'actor_{name}_states')
        self.q = tf.placeholder(shape=None, dtype=tf.float32, name=f'actor_{name}_q')
        self.actions = tf.placeholder(shape=None, dtype=tf.float32, name=f'actor_{name}_actions')
        
        # ネットワーク構築
        _layer = self.states
        for i, layer_size in enumerate(hidden_layers_size):
            _layer = tf.layers.dense(
                inputs=_layer, 
                units=layer_size, 
                activation=tf.nn.relu,
                kernel_initializer=tf2.keras.initializers.GlorotNormal(),
                name=f'actor_{name}_layer_{i}'
            )
        
        self.predictions = tf.layers.dense(
            inputs=_layer, 
            units=1, 
            activation=tf.nn.sigmoid,  # 0-1の範囲に正規化
            kernel_initializer=tf2.keras.initializers.GlorotNormal(),
            name=f'actor_{name}_output'
        )
        
        # 出力を実際のアクション範囲にスケール
        self.scaled_predictions = min_value + self.predictions * (max_value - min_value)
        
        # 方策勾配損失
        self.cost = -tf.reduce_mean(self.q * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

# 経験再生メモリ
class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)
        
    def __len__(self):
        return len(self.memory)
    
    def append(self, element):
        self.memory.append(element)
    
    def sample(self, n):
        if n > len(self.memory):
            n = len(self.memory)
        return random.sample(self.memory, n)

# Ornstein-Uhlenbeck ノイズ
class OUNoise:
    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.x = self.mu
    
    def noise(self):
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.normal()
        self.x += dx
        return self.x

# Unity用A2Cエージェント
class UnityGraspA2C:
    def __init__(self, input_size, memory_size, actor_hidden_layers, critic_hidden_layers,
                 actor_gamma, critic_gamma, actor_lr, critic_lr, min_action, max_action,
                 tau=1.0, epochs_to_copy=1000, ou_mu=0, ou_theta=0.15, ou_sigma=0.2):
        
        # ネットワーク初期化
        self.critic = QNetwork(critic_hidden_layers, critic_gamma, critic_lr, input_size, 'online')
        self.actor = PolicyGradient(actor_hidden_layers, actor_gamma, actor_lr, input_size,
                                   min_action, max_action, 'online')
        
        self.critic_target = QNetwork(critic_hidden_layers, critic_gamma, critic_lr, input_size, 'target')
        self.actor_target = PolicyGradient(actor_hidden_layers, actor_gamma, actor_lr, input_size,
                                          min_action, max_action, 'target')
        
        self.memory = ReplayMemory(memory_size)
        self.noise = OUNoise(ou_mu, ou_theta, ou_sigma)
        
        self.tau = tau
        self.epochs_to_copy = epochs_to_copy
        self.epoch_counter = 0
        
    def _extract_from_batch(self, batch, key):
        return np.array([x[key] for x in batch])
    
    def _update_terminal_states(self, q_target, terminals):
        for i in range(len(terminals)):
            if terminals[i]:
                q_target[i] = 0.0
        return q_target
    
    def _copy_networks(self, session):
        """ターゲットネットワークの更新"""
        tf_vars = tf.trainable_variables()
        num_vars = len(tf_vars)
        operations = []
        
        for i, v in enumerate(tf_vars[0:num_vars // 2]):
            target_var = tf_vars[i + num_vars // 2]
            operations.append(target_var.assign(
                (v.value() * self.tau) + ((1 - self.tau) * target_var.value())))
        
        session.run(operations)
    
    def reset_noise(self):
        self.noise.reset()
    
    def remember(self, **kwargs):
        self.memory.append(kwargs)
    
    def act(self, session, state, greedy=False):
        """アクション選択"""
        action = session.run(self.actor.scaled_predictions,
                           feed_dict={self.actor.states: np.expand_dims(state, axis=0)}).flatten()
        
        if not greedy:
            # 探索ノイズを追加
            action += abs(self.noise.noise())
        
        return action
    
    def learn(self, session, batch_size):
        """学習実行"""
        self.epoch_counter += 1
        
        if len(self.memory) < batch_size:
            return 0, 0
        
        # バッチサンプリング
        batch = self.memory.sample(batch_size)
        
        states = self._extract_from_batch(batch, 'state')
        actions = self._extract_from_batch(batch, 'action')
        rewards = self._extract_from_batch(batch, 'reward')
        next_states = self._extract_from_batch(batch, 'next_state')
        terminals = self._extract_from_batch(batch, 'done')
        
        # Q値計算
        q_next = session.run(self.critic_target.predictions,
                           feed_dict={self.critic_target.states: next_states})
        q_next = self._update_terminal_states(q_next.flatten(), terminals)
        
        # Criticの学習
        critic_cost, _ = session.run([self.critic.cost, self.critic.optimizer],
                                   feed_dict={
                                       self.critic.states: states,
                                       self.critic.r: rewards,
                                       self.critic.q_target: q_next.reshape(-1, 1),
                                       self.critic.actions: actions.reshape(-1, 1)
                                   })
        
        # Actorの学習用Q値取得
        q_values = session.run(self.critic.predictions,
                             feed_dict={self.critic.states: states})
        
        # Actorの学習
        actor_cost, _ = session.run([self.actor.cost, self.actor.optimizer],
                                  feed_dict={
                                      self.actor.states: states,
                                      self.actor.q: q_values.flatten(),
                                      self.actor.actions: actions
                                  })
        
        # ターゲットネットワーク更新
        if self.epoch_counter % self.epochs_to_copy == 0:
            self._copy_networks(session)
        
        return actor_cost, critic_cost

# メイン実行部分
def main():
    # Unity-ROS接続
    unity_interface = UnityROSInterface()
    
    # 環境設定
    env = UnityGraspEnvironment(unity_interface)
    
    # ハイパーパラメータ
    actor_hidden_layers = [64, 32]
    critic_hidden_layers = [64, 64, 32]
    actor_learning_rate = 0.001
    critic_learning_rate = 0.002
    gamma = 0.99
    batch_size = 32
    memory_size = 1000
    
    # A2Cエージェント初期化
    agent = UnityGraspA2C(
        input_size=env.state_size,
        memory_size=memory_size,
        actor_hidden_layers=actor_hidden_layers,
        critic_hidden_layers=critic_hidden_layers,
        actor_gamma=gamma,
        critic_gamma=gamma,
        actor_lr=actor_learning_rate,
        critic_lr=critic_learning_rate,
        min_action=env.action_space_low,
        max_action=env.action_space_high
    )
    
    # TensorFlowセッション開始
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # 学習ループ
    episodes = 200
    results_df = pd.DataFrame(columns=['episode', 'steps', 'total_reward', 'actor_cost', 'critic_cost'])
    
    print("Unity-ROS A2C学習開始...")
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset_noise()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # アクション選択
            action = agent.act(sess, state)
            
            # 環境でステップ実行
            next_state, reward, done, _ = env.step(action)
            
            # 経験保存
            agent.remember(
                state=state,
                action=action[0],
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # 学習実行
            actor_cost, critic_cost = agent.learn(sess, batch_size)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if steps % 10 == 0:
                print(f"Episode {episode+1}, Step {steps}, Reward: {reward:.2f}, "
                      f"Action: {action[0]:.2f}, Total: {total_reward:.2f}")
        
        # エピソード結果記録
        results_df = results_df.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': total_reward,
            'actor_cost': actor_cost,
            'critic_cost': critic_cost
        }, ignore_index=True)
        
        print(f"Episode {episode+1} 完了: Steps={steps}, Total Reward={total_reward:.2f}")
        
        # 定期的な結果保存
        if (episode + 1) % 50 == 0:
            results_df.to_csv(f'unity_a2c_results_ep{episode+1}.csv', index=False)
    
    # 最終結果保存
    results_df.to_csv('unity_a2c_final_results.csv', index=False)
    
    # 学習済みモデルでテスト実行
    print("\n学習済みモデルでテスト実行...")
    test_episodes = 5
    
    for test_ep in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nテストエピソード {test_ep+1}:")
        
        while not done and steps < 100:
            action = agent.act(sess, state, greedy=True)  # 探索なし
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            print(f"  Step {steps}: Action={action[0]:.2f}, Reward={reward:.2f}")
        
        print(f"テストエピソード {test_ep+1} 結果: {total_reward:.2f} (Steps: {steps})")

if __name__ == "__main__":
    main()