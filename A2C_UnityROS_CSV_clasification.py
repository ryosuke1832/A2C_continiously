# grasp_force_a2c_tf2_with_csv.py
import socket
import json
import threading
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# TensorFlow 2.x設定
tf.config.run_functions_eagerly(False)

class GraspForceA2CAgent:
    """把持力制御用A2Cエージェント（TensorFlow 2.x対応）"""
    
    def __init__(self, state_size=4, action_size=1, 
                 actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        
        self.state_size = state_size  # [current_force, accumulated_force, is_crushed, time_step]
        self.action_size = action_size  # [target_force_adjustment]
        self.gamma = gamma
        
        # ネットワーク構築
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # オプティマイザ
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        
        # 経験バッファ
        self.memory = deque(maxlen=2000)
        
        # 制御パラメータ
        self.force_range = (0.0, 25.0)  # 制御する力の範囲
        self.safe_force_range = (2.0, 15.0)  # 安全な力の範囲
        self.episode_start_time = None  # リセット時に設定されるように変更
            
        print("🤖 把持力制御A2Cエージェント初期化完了（TensorFlow 2.x）")
    
    def _build_actor(self):
        """アクターネットワーク構築"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')  # 力の調整値: -1.0 ~ +1.0
        ], name='actor')
        
        return model
    
    def _build_critic(self):
        """クリティックネットワーク構築"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=None)  # 状態価値
        ], name='critic')
        
        return model
    
    def get_action(self, state, exploration=True):
        """状態から行動を決定"""
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        
        # アクターから力調整値を取得
        force_adjustment = float(self.actor(state)[0][0].numpy())
        
        # 探索ノイズ追加
        if exploration:
            noise = np.random.normal(0, 0.1)
            force_adjustment += noise
            force_adjustment = float(np.clip(force_adjustment, -1.0, 1.0))
        
        return force_adjustment
    
    def get_value(self, state):
        """状態価値を取得"""
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        value = float(self.critic(state)[0][0].numpy())
        return value
    
    def remember(self, state, action, reward, next_state, done):
        """経験を記憶"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """単一の訓練ステップ"""
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # 現在の状態価値と次の状態価値を計算
            current_values = self.critic(states)
            next_values = self.critic(next_states)
            
            # ターゲット価値計算（TD学習）
            target_values = rewards + self.gamma * next_values * (1 - dones)
            advantages = target_values - current_values
            
            # クリティック損失
            critic_loss = tf.reduce_mean(tf.square(advantages))
            
            # アクター損失（ポリシー勾配）
            action_probs = self.actor(states)
            log_probs = -0.5 * tf.square((actions - action_probs) / 0.1)
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
        
        # 勾配計算と適用
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        return actor_loss, critic_loss
    
    def train(self, batch_size=32):
        """ネットワークを訓練"""
        if len(self.memory) < batch_size:
            return None, None
        
        # バッチサンプリング
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        states = tf.convert_to_tensor([exp['state'] for exp in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([[exp['action']] for exp in batch], dtype=tf.float32)
        rewards = tf.convert_to_tensor([[exp['reward']] for exp in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([exp['next_state'] for exp in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([[float(exp['done'])] for exp in batch], dtype=tf.float32)
        
        # 訓練実行
        actor_loss, critic_loss = self.train_step(states, actions, rewards, next_states, dones)
        
        return float(actor_loss.numpy()), float(critic_loss.numpy())

class GraspForceA2CServer:
    """Unity通信対応の把持力制御A2Cサーバー（CSV保存機能付き）"""
    
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.running = False
        
        # A2Cエージェント
        self.agent = GraspForceA2CAgent()
        
        # エピソード管理
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
        # 状態管理
        self.current_state = None
        self.last_action = 0.0
        self.last_recommended_force = 10.0  # 初期推奨力
        
        # 統計
        self.total_episodes = 0
        self.successful_episodes = 0
        self.current_episode_reward = 0.0
        
        # CSV保存用データフレーム
        self.results_df = pd.DataFrame(columns=['game', 'steps', 'actor_cost', 'critic_cost', 'final_reward', 'action'])
        
        # エピソードごとの詳細データ
        self.episode_actions = []
        self.episode_actor_costs = []
        self.episode_critic_costs = []
        
        print("🚀 把持力制御A2Cサーバー初期化完了（CSV保存機能付き）")
    
    def start_server(self):
        """サーバー開始"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(1.0)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            print("=" * 60)
            print("🤖 把持力制御A2Cサーバー開始")
            print(f"📡 接続先: {self.host}:{self.port}")
            print("🎯 目標: アルミ缶をつぶさない最適な把持力学習")
            print("💾 結果をCSVファイルに保存します")
            print("=" * 60)
            
            self.running = True
            
            while self.running:
                try:
                    self.client_socket, addr = self.socket.accept()
                    print(f"✅ Unity接続: {addr}")
                    print(f"⏰ 開始時刻: {datetime.now().strftime('%H:%M:%S')}")
                    
                    self.communication_loop()
                    break
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"❌ 接続エラー: {e}")
                    break
        
        except Exception as e:
            print(f"❌ サーバー開始エラー: {e}")
        finally:
            self.cleanup()
    
        
    def communication_loop(self):
        """メイン通信ループ"""
        print("🧠 A2C学習ループ開始")
        print("📊 リアルタイム学習データを表示中...")
        print()
        
        if self.client_socket:
            self.client_socket.settimeout(1.0)
        
        while self.running:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break
                
                # 🔥 追加：受信データをそのまま表示
                raw_message = data.decode('utf-8')
                print(f"📥 受信データ（生データ）: {raw_message}")
                
                message = json.loads(raw_message)
                
                # 🔥 追加：JSONパース後の内容も表示
                print(f"📋 パース後の内容: {json.dumps(message, indent=2, ensure_ascii=False)}")
                
                response = self.process_a2c_message(message)
                
                if response:
                    response_json = json.dumps(response)
                    # 🔥 追加：送信するレスポンス内容も表示
                    print(f"📤 送信データ: {response_json}")
                    self.client_socket.send(response_json.encode('utf-8'))
                
            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                print(f"❌ JSONError: {e}")
                print(f"🔍 エラーの原因データ: {data.decode('utf-8') if data else 'None'}")  # 🔥 追加
            except Exception as e:
                if self.running:
                    print(f"❌ Communication Error: {e}")
                    print(f"🔍 エラー時のデータ: {data.decode('utf-8') if data else 'None'}")  # 🔥 追加
                break
    
    def process_a2c_message(self, message):
        """A2C学習メッセージ処理"""
        msg_type = message.get('type')
        
        # 🔥 追加：メッセージタイプと基本情報を表示
        print(f"🎯 メッセージ処理開始: タイプ={msg_type}, タイムスタンプ={message.get('timestamp', 'なし')}")
        
        if msg_type == 'can_state':
            # 🔥 追加：缶の状態詳細を表示
            current_force = message.get('current_force', 0.0)
            accumulated_force = message.get('accumulated_force', 0.0)
            is_crushed = message.get('is_crushed', False)
            print(f"🥤 缶状態: 現在力={current_force:.2f}N, 累積力={accumulated_force:.2f}N, つぶれ={is_crushed}")
            
            return self.handle_can_state_learning(message)
        elif msg_type == 'episode_end':
            print(f"🏁 エピソード終了メッセージ受信")
            return self.handle_episode_end(message)
        elif msg_type == 'reset':
            print(f"🔄 リセットメッセージ受信")
            return self.handle_reset(message)
        elif msg_type == 'ping':
            print(f"🏓 Pingメッセージ受信")
            return {
                'type': 'pong', 
                'message': 'A2Cエージェントが動作中',
                'timestamp': float(time.time())
            }
        else:
            # 🔥 追加：不明なメッセージタイプの詳細表示
            print(f"❓ 不明なメッセージタイプ: {msg_type}")
            print(f"🔍 メッセージ全体: {json.dumps(message, indent=2, ensure_ascii=False)}")
        
        return None

    def communication_loop(self):
        print("🧠 A2C学習ループ開始")
        print("📊 リアルタイム学習データを表示中...")
        print()
        
        if self.client_socket:
            self.client_socket.settimeout(1.0)
        
        while self.running:
            try:
                data = self.client_socket.recv(4096)
                if not data:
                    break
                
                # 🔥 追加：受信データをそのまま表示
                raw_message = data.decode('utf-8')
                print(f"📥 受信データ（生データ）: {raw_message}")
                
                message = json.loads(raw_message)
                
                # 🔥 追加：JSONパース後の内容も表示
                print(f"📋 パース後の内容: {json.dumps(message, indent=2, ensure_ascii=False)}")
                
                response = self.process_a2c_message(message)
                
                if response:
                    response_json = json.dumps(response)
                    # 🔥 追加：送信するレスポンス内容も表示
                    print(f"📤 送信データ: {response_json}")
                    self.client_socket.send(response_json.encode('utf-8'))
                
            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                print(f"❌ JSONError: {e}")
                print(f"🔍 エラーの原因データ: {data.decode('utf-8') if data else 'None'}")  # 🔥 追加
            except Exception as e:
                if self.running:
                    print(f"❌ Communication Error: {e}")
                    print(f"🔍 エラー時のデータ: {data.decode('utf-8') if data else 'None'}")  # 🔥 追加
                break

    def process_a2c_message(self, message):
        """A2C学習メッセージ処理"""
        msg_type = message.get('type')
        
        # 🔥 追加：メッセージタイプと基本情報を表示
        print(f"🎯 メッセージ処理開始: タイプ={msg_type}, タイムスタンプ={message.get('timestamp', 'なし')}")
        
        if msg_type == 'can_state':
            # 🔥 追加：缶の状態詳細を表示
            current_force = message.get('current_force', 0.0)
            accumulated_force = message.get('accumulated_force', 0.0)
            is_crushed = message.get('is_crushed', False)
            print(f"🥤 缶状態: 現在力={current_force:.2f}N, 累積力={accumulated_force:.2f}N, つぶれ={is_crushed}")
            
            return self.handle_can_state_learning(message)
        elif msg_type == 'episode_end':
            print(f"🏁 エピソード終了メッセージ受信")
            return self.handle_episode_end(message)
        elif msg_type == 'reset':
            print(f"🔄 リセットメッセージ受信")
            return self.handle_reset(message)
        elif msg_type == 'ping':
            print(f"🏓 Pingメッセージ受信")
            return {
                'type': 'pong', 
                'message': 'A2Cエージェントが動作中',
                'timestamp': float(time.time())
            }
        else:
            # 🔥 追加：不明なメッセージタイプの詳細表示
            print(f"❓ 不明なメッセージタイプ: {msg_type}")
            print(f"🔍 メッセージ全体: {json.dumps(message, indent=2, ensure_ascii=False)}")
        
        return None
    
        
    def handle_can_state_learning(self, message):
        """把持状態での学習処理"""
        # 🔥 追加：受信したメッセージの詳細ログ
        print(f"📊 学習データ受信:")
        for key, value in message.items():
            print(f"   {key}: {value}")
        
        # 現在の状態を構築
        current_force = message.get('current_force', 0.0)
        accumulated_force = message.get('accumulated_force', 0.0)
        is_crushed = message.get('is_crushed', False)
        timestamp = message.get('timestamp', time.time())
        
        # 🔥 修正：エピソード開始時間の初期化チェック
        if self.episode_start_time is None:
            self.episode_start_time = timestamp
            print(f"⚠️ エピソード開始時間が未設定だったため、現在時間で初期化: {timestamp:.3f}秒")
        
        # 🔥 修正：時間正規化の計算を改善
        elapsed_time = timestamp - self.episode_start_time
        time_step = elapsed_time / 30.0  # 30秒で正規化
        
        # 🔥 追加：時間計算のデバッグログ
        print(f"⏰ 時間計算: 現在時間={timestamp:.3f}, 開始時間={self.episode_start_time:.3f}, 経過時間={elapsed_time:.3f}, 正規化値={time_step:.6f}")
        
        # 正規化された状態ベクトル
        new_state = [
            current_force / 25.0,  # 25Nで正規化
            accumulated_force / 25.0,
            float(is_crushed),
            min(max(time_step, 0.0), 1.0)  # 🔥 修正：0-1の範囲にクリップ
        ]
        
        # 🔥 修正：正規化後の状態ベクトルも表示
        print(f"🧮 正規化状態ベクトル: {new_state}")
        
        # 報酬計算
        reward = self.calculate_grasp_reward(current_force, accumulated_force, is_crushed)
        self.current_episode_reward += reward
        
        # 🔥 追加：報酬計算の詳細
        print(f"💰 計算された報酬: {reward:.3f}, 累積報酬: {self.current_episode_reward:.3f}")
        
        
        # 損失値の初期化
        actor_loss, critic_loss = 0.0, 0.0
        
        # 学習処理（前の状態がある場合）
        if self.current_state is not None:
            done = is_crushed  # つぶれたらエピソード終了
            
            # 経験を保存
            self.agent.remember(
                self.current_state, 
                self.last_action, 
                reward, 
                new_state, 
                done
            )
            
            # 学習実行
            if len(self.agent.memory) > 32:
                actor_loss, critic_loss = self.agent.train()
                
                # 損失値を記録（学習が実行された場合のみ）
                if actor_loss is not None and critic_loss is not None:
                    self.episode_actor_costs.append(actor_loss)
                    self.episode_critic_costs.append(critic_loss)
                
                if actor_loss is not None and self.episode_steps % 10 == 0:  # 10ステップごとに表示
                    print(f"📈 Episode {self.current_episode}, Step {self.episode_steps}: "
                          f"報酬={reward:.2f}, "
                          f"現在力={current_force:.2f}N, "
                          f"Actor損失={actor_loss:.4f}, "
                          f"Critic損失={critic_loss:.4f}")
        
        # 次の行動を決定
        action = self.agent.get_action(new_state, exploration=True)
        
        # 行動を推奨力に変換
        recommended_force = self.action_to_force(action, current_force)
        
        # アクションを記録
        self.episode_actions.append(recommended_force)
        
        # 状態更新
        self.current_state = new_state
        self.last_action = action
        self.last_recommended_force = recommended_force
        self.episode_steps += 1
        
        # エピソード終了チェック
        if is_crushed or self.episode_steps > 100:  # つぶれるか100ステップで終了
            self.end_current_episode(is_crushed)
        
        return {
            'type': 'a2c_action',
            'recommended_force': float(recommended_force),
            'current_reward': float(reward),
            'episode': int(self.current_episode),
            'step': int(self.episode_steps),
            'exploration_factor': float(action),
            'timestamp': float(time.time())
        }
    
    def calculate_grasp_reward(self, current_force, accumulated_force, is_crushed):
        """把持報酬計算"""
        if is_crushed:
            return -100.0  # つぶれたら大きなペナルティ
        
        # 基本報酬：適切な力範囲にいることに対する報酬
        if 2.0 <= current_force <= 15.0:
            # 最適範囲での報酬（中央付近で最大）
            optimal_force = 8.5  # 最適力
            distance_from_optimal = abs(current_force - optimal_force)
            force_reward = 10.0 * (1.0 - distance_from_optimal / 6.5)  # 最大10ポイント
        elif current_force < 2.0:
            # 力不足のペナルティ
            force_reward = -5.0
        else:  # current_force > 15.0
            # 過度な力のペナルティ
            excess_force = current_force - 15.0
            force_reward = -excess_force * 2.0  # 過剰力に比例したペナルティ
        
        # 継続報酬：つぶさずに維持できている時間に対する報酬
        time_reward = 1.0  # 各ステップで1ポイント
        
        # 蓄積力に対する注意喚起
        if accumulated_force > 12.0:
            accumulation_penalty = -(accumulated_force - 12.0) * 0.5
        else:
            accumulation_penalty = 0.0
        
        total_reward = force_reward + time_reward + accumulation_penalty
        
        return total_reward
    
    def action_to_force(self, action, current_force):
        """行動値を推奨力に変換"""
        # actionは-1.0~1.0の範囲
        # 現在の力から±3Nの範囲で調整
        force_adjustment = float(action) * 3.0  # 最大±3Nの調整
        new_force = float(current_force) + force_adjustment
        
        # 安全範囲にクリップ
        new_force = float(np.clip(new_force, 0.0, 20.0))
        
        return new_force
    
    def end_current_episode(self, was_crushed):
        """エピソード終了処理とCSV保存"""
        self.total_episodes += 1
        self.episode_rewards.append(self.current_episode_reward)
        
        if not was_crushed:
            self.successful_episodes += 1
        
        # エピソードの統計値計算
        avg_action = np.mean(self.episode_actions) if self.episode_actions else 0.0
        avg_actor_cost = np.mean(self.episode_actor_costs) if self.episode_actor_costs else 0.0
        avg_critic_cost = np.mean(self.episode_critic_costs) if self.episode_critic_costs else 0.0
        
        # CSVデータに追加
        episode_data = {
            'game': self.current_episode,
            'steps': self.episode_steps,
            'actor_cost': avg_actor_cost,
            'critic_cost': avg_critic_cost,
            'final_reward': self.current_episode_reward,
            'action': avg_action
        }
        
        # DataFrameに行を追加
        new_row = pd.DataFrame([episode_data])
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # 進行統計の計算
        success_rate = (self.successful_episodes / self.total_episodes) * 100
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        
        print(f"\n🏁 エピソード {self.current_episode} 終了")
        print(f"   結果: {'✅成功' if not was_crushed else '❌失敗（つぶれた）'}")
        print(f"   ステップ数: {self.episode_steps}")
        print(f"   エピソード報酬: {self.current_episode_reward:.2f}")
        print(f"   平均アクション: {avg_action:.2f}N")
        print(f"   平均Actor損失: {avg_actor_cost:.4f}")
        print(f"   平均Critic損失: {avg_critic_cost:.4f}")
        print(f"   成功率: {success_rate:.1f}% ({self.successful_episodes}/{self.total_episodes})")
        print(f"   平均報酬(直近10回): {avg_reward:.2f}")
        print("-" * 50)
        
        # 定期的にCSVファイルを保存
        if self.current_episode % 10 == 0:
            self.save_results_csv()
        
        # 次のエピソード準備
        self.current_episode += 1
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.current_state = None
        self.current_episode_reward = 0.0
        
        # エピソードデータをリセット
        self.episode_actions = []
        self.episode_actor_costs = []
        self.episode_critic_costs = []
    
    def save_results_csv(self):
        """結果をCSVファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'grasp_a2c_results_{timestamp}.csv'
        
        try:
            self.results_df.to_csv(filename, index=False)
            print(f"💾 結果をCSVファイルに保存しました: {filename}")
        except Exception as e:
            print(f"❌ CSV保存エラー: {e}")
    
    def save_final_results(self):
        """最終結果とグラフを保存"""
        if len(self.results_df) == 0:
            print("⚠️ 保存するデータがありません")
            return
        
        # 最終CSVファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'grasp_a2c_final_results_{timestamp}.csv'
        
        try:
            self.results_df.to_csv(csv_filename, index=False)
            print(f"💾 最終結果をCSVファイルに保存しました: {csv_filename}")
            
            # 学習曲線の可視化
            self.plot_learning_curves(timestamp)
            
        except Exception as e:
            print(f"❌ 最終結果保存エラー: {e}")
    
    def plot_learning_curves(self, timestamp):
        """学習曲線をプロット"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 移動平均のウィンドウサイズ
            window_size = min(10, len(self.results_df))
            
            if window_size > 0:
                # ステップ数の移動平均
                steps_ma = self.results_df['steps'].rolling(window=window_size).mean()
                axes[0,0].plot(self.results_df['game'], steps_ma)
                axes[0,0].set_title('Steps per Episode (Moving Average)')
                axes[0,0].set_xlabel('Episode')
                axes[0,0].set_ylabel('Steps')
                axes[0,0].grid(True)
                
                # 報酬の移動平均
                reward_ma = self.results_df['final_reward'].rolling(window=window_size).mean()
                axes[0,1].plot(self.results_df['game'], reward_ma)
                axes[0,1].set_title('Final Reward per Episode (Moving Average)')
                axes[0,1].set_xlabel('Episode')
                axes[0,1].set_ylabel('Final Reward')
                axes[0,1].grid(True)
                
                # Actor損失の移動平均
                actor_cost_ma = self.results_df['actor_cost'].rolling(window=window_size).mean()
                axes[1,0].plot(self.results_df['game'], actor_cost_ma)
                axes[1,0].set_title('Actor Cost (Moving Average)')
                axes[1,0].set_xlabel('Episode')
                axes[1,0].set_ylabel('Actor Cost')
                axes[1,0].grid(True)
                
                # 平均アクションの移動平均
                action_ma = self.results_df['action'].rolling(window=window_size).mean()
                axes[1,1].plot(self.results_df['game'], action_ma)
                axes[1,1].set_title('Average Action per Episode (Moving Average)')
                axes[1,1].set_xlabel('Episode')
                axes[1,1].set_ylabel('Average Grip Force (N)')
                axes[1,1].grid(True)
            
            plt.tight_layout()
            plot_filename = f'grasp_a2c_learning_curves_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 学習曲線を保存しました: {plot_filename}")
            
            # 統計情報を表示
            self.print_final_statistics()
            
        except Exception as e:
            print(f"❌ グラフ作成エラー: {e}")
    
    def print_final_statistics(self):
        """最終統計情報を表示"""
        if len(self.results_df) > 0:
            print("\n" + "=" * 60)
            print("📊 最終学習統計")
            print("-" * 60)
            print(f"総エピソード数: {len(self.results_df)}")
            print(f"平均ステップ数: {self.results_df['steps'].mean():.2f}")
            print(f"平均最終報酬: {self.results_df['final_reward'].mean():.2f}")
            print(f"平均Actor損失: {self.results_df['actor_cost'].mean():.4f}")
            print(f"平均Critic損失: {self.results_df['critic_cost'].mean():.4f}")
            print(f"平均アクション: {self.results_df['action'].mean():.2f}N")
            
            # 直近10エピソードの統計
            if len(self.results_df) >= 10:
                recent_data = self.results_df.tail(10)
                print(f"\n直近10エピソードの統計:")
                print(f"平均ステップ数: {recent_data['steps'].mean():.2f}")
                print(f"平均最終報酬: {recent_data['final_reward'].mean():.2f}")
                print(f"平均アクション: {recent_data['action'].mean():.2f}N")
            
            print("=" * 60)
    
    def handle_episode_end(self, message):
        """明示的なエピソード終了処理"""
        if self.current_state is not None:
            self.end_current_episode(False)  # 正常終了
        
        return {
            'type': 'episode_ack',
            'total_episodes': int(self.total_episodes),
            'success_rate': float((self.successful_episodes / max(1, self.total_episodes)) * 100),
            'timestamp': float(time.time())
        }
        
    def handle_reset(self, message):
        """リセット処理"""
        self.current_state = None
        self.episode_steps = 0
        
        # 🔥 修正：Unity時間に合わせたエピソード開始時間の設定
        unity_timestamp = message.get('timestamp', time.time())
        self.episode_start_time = unity_timestamp  # Unity時間を使用
        
        # 🔥 追加：デバッグログ
        print(f"🔄 リセット処理：Unity時間={unity_timestamp:.3f}秒をエピソード開始時間として設定")
        
        self.current_episode_reward = 0.0
        self.episode_actions = []
        self.episode_actor_costs = []
        self.episode_critic_costs = []
        
        return {
            'type': 'reset_ack',  
            'message': 'A2Cエージェントリセット完了',
            'timestamp': unity_timestamp 
        }
    
    def cleanup(self):
        """クリーンアップ"""
        self.running = False
        
        # 最終結果を保存
        self.save_final_results()
        
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        
        # 最終統計表示
        print("\n" + "=" * 60)
        print("🏆 A2C学習統計")
        print("-" * 60)
        print(f"総エピソード数: {self.total_episodes}")
        print(f"成功エピソード数: {self.successful_episodes}")
        if self.total_episodes > 0:
            print(f"最終成功率: {(self.successful_episodes / self.total_episodes) * 100:.2f}%")
            if self.episode_rewards:
                print(f"平均報酬: {np.mean(self.episode_rewards):.2f}")
                print(f"最高報酬: {max(self.episode_rewards):.2f}")
        print("=" * 60)
        print("🛑 A2Cサーバー停止")

# メイン実行
if __name__ == "__main__":
    print("把持力制御A2Cエージェント（TensorFlow 2.x対応・CSV保存機能付き）")
    print("Unity連携による強化学習システム")
    print()
    
    server = GraspForceA2CServer()
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n⏹️ 手動停止中...")
        server.cleanup()