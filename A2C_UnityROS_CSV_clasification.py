# refactored_force_boundary_classifier.py
# 元のForceBoundaryClassifierを分離したTCP通信部分を使ってリファクタリング

import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
import os
from typing import Dict, Any

# Warning抑制設定
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# TensorFlow警告をさらに抑制
tf.get_logger().setLevel('ERROR')

# 分離したTCP通信部分をインポート
from utility.tcp_communication import UnityTcpServer

class ForceBoundaryClassifier:
    """
    適切な力加減の境界線を判別するAIシステム（リファクタリング版）
    通信部分を分離してビジネスロジックに集中
    """
    
    def __init__(self):
        # 分類モデル用パラメータ
        self.scaler = StandardScaler()
        self.rf_classifier = None
        self.nn_classifier = None
        self.training_data = []
        self.labels = []
        
        # データ収集用変数
        self.session_data = []
        self.current_episode = 0
        self.current_step = 0
        self.start_time = time.time()
        
        # 把持力の境界値設定
        self.safe_force_threshold = 15.0  # N - 安全な把持力の上限
        self.minimum_grip_threshold = 5.0  # N - 把持成功の最小力
        self.critical_force_threshold = 25.0  # N - 缶が潰れる危険領域
        
        # 分類ラベル定義
        self.FORCE_CATEGORIES = {
            'under_grip': 0,      # 把持力不足（滑り落ちる）
            'optimal_grip': 1,    # 最適な把持力（成功）
            'over_grip': 2,       # 把持力過多（潰れる危険）
            'critical_force': 3   # 危険な力（確実に潰れる）
        }
        
        print("🤖 Force Boundary Classifier初期化完了")
        print(f"📊 分類カテゴリ: {list(self.FORCE_CATEGORIES.keys())}")
        print(f"⚖️  安全閾値: {self.safe_force_threshold}N")
        print(f"🎯 最適範囲: {self.minimum_grip_threshold}N - {self.safe_force_threshold}N")

    def classify_force_level(self, current_force, accumulated_force, is_crushed):
        """
        現在の把持力レベルを分類する
        """
        if is_crushed:
            return self.FORCE_CATEGORIES['critical_force']
        elif current_force >= self.critical_force_threshold:
            return self.FORCE_CATEGORIES['critical_force']
        elif current_force >= self.safe_force_threshold:
            return self.FORCE_CATEGORIES['over_grip']
        elif current_force >= self.minimum_grip_threshold:
            return self.FORCE_CATEGORIES['optimal_grip']
        else:
            return self.FORCE_CATEGORIES['under_grip']

    def extract_features(self, data):
        """
        把持状態から特徴量を抽出
        """
        features = [
            data['current_force'],
            data['accumulated_force'],
            data['timestamp'],
            int(data['is_crushed']),
            # 時系列特徴量（前の状態との差分など）
            data.get('force_change_rate', 0.0),
            data.get('time_since_start', 0.0)
        ]
        return np.array(features)

    def create_neural_network_classifier(self, input_dim):
        """
        ニューラルネットワーク分類器を作成（警告解消版）
        """
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(len(self.FORCE_CATEGORIES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_classifiers(self):
        """
        分類器を訓練する
        """
        if len(self.training_data) < 50:
            print("⚠️  訓練データが不足しています（最小50サンプル必要）")
            return False
        
        # データの準備
        X = np.array(self.training_data)
        y = np.array(self.labels)
        
        # データの標準化
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練・テストデータの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest分類器の訓練
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_classifier.fit(X_train, y_train)
        
        # ニューラルネットワーク分類器の訓練
        self.nn_classifier = self.create_neural_network_classifier(X_train.shape[1])
        self.nn_classifier.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # 性能評価
        rf_predictions = self.rf_classifier.predict(X_test)
        nn_predictions = np.argmax(self.nn_classifier.predict(X_test), axis=1)
        
        print("🎯 Random Forest分類器性能:")
        print(f"   精度: {accuracy_score(y_test, rf_predictions):.3f}")
        
        print("🧠 Neural Network分類器性能:")
        print(f"   精度: {accuracy_score(y_test, nn_predictions):.3f}")
        
        return True

    def predict_force_category(self, features):
        """
        把持力のカテゴリを予測
        """
        if self.rf_classifier is None or self.nn_classifier is None:
            return None
        
        features_scaled = self.scaler.transform([features])
        
        # Random Forestによる予測
        rf_prediction = self.rf_classifier.predict(features_scaled)[0]
        rf_probability = self.rf_classifier.predict_proba(features_scaled)[0]
        
        # Neural Networkによる予測
        nn_prediction = np.argmax(self.nn_classifier.predict(features_scaled))
        nn_probability = self.nn_classifier.predict(features_scaled)[0]
        
        return {
            'rf_prediction': rf_prediction,
            'rf_confidence': max(rf_probability),
            'nn_prediction': nn_prediction,
            'nn_confidence': max(nn_probability),
            'category_name': list(self.FORCE_CATEGORIES.keys())[rf_prediction]
        }

    def generate_force_recommendation(self, current_state, prediction):
        """
        現在の状態に基づいて推奨力を生成
        """
        current_force = current_state['current_force']
        category = prediction['category_name']
        
        if category == 'under_grip':
            # 力が不足している場合、安全範囲内で増加
            recommended_force = min(current_force + 2.0, self.safe_force_threshold)
            action = "力を増加"
        elif category == 'optimal_grip':
            # 最適範囲の場合、現状維持
            recommended_force = current_force
            action = "現状維持"
        elif category == 'over_grip':
            # 力が強すぎる場合、減少
            recommended_force = max(current_force - 1.5, self.minimum_grip_threshold)
            action = "力を減少"
        else:  # critical_force
            # 危険レベルの場合、大幅減少
            recommended_force = self.minimum_grip_threshold
            action = "緊急減力"
        
        return {
            'recommended_force': recommended_force,
            'action': action,
            'safety_level': category,
            'confidence': prediction['rf_confidence']
        }

    def save_session_data(self, data_row):
        """
        セッションデータをCSVファイルに保存
        """
        self.session_data.append(data_row)
        
        # 100ステップごとにCSVファイルに保存
        if len(self.session_data) % 100 == 0:
            df = pd.DataFrame(self.session_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"force_boundary_classification_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 分類結果をCSVファイルに保存: {filename}")

    def handle_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        メッセージハンドラー（通信部分から分離されたビジネスロジック）
        """
        if message_type == 'ping':
            # Ping応答
            return {
                "type": "pong",
                "message": "Force Boundary Classifier動作中",
                "timestamp": time.time()
            }
            
        elif message_type == 'can_state':
            # 缶の状態データを処理
            current_force = data['current_force']
            accumulated_force = data['accumulated_force']
            is_crushed = data['is_crushed']
            timestamp = data['timestamp']
            
            # 分類ラベル生成
            force_category = self.classify_force_level(
                current_force, accumulated_force, is_crushed
            )
            
            # 特徴量抽出
            features = self.extract_features({
                'current_force': current_force,
                'accumulated_force': accumulated_force,
                'timestamp': timestamp,
                'is_crushed': is_crushed,
                'force_change_rate': 0.0,  # TODO: 前の状態との比較
                'time_since_start': time.time() - self.start_time
            })
            
            # 訓練データとして蓄積
            self.training_data.append(features)
            self.labels.append(force_category)
            
            # 予測実行（分類器が訓練済みの場合）
            prediction = None
            if len(self.training_data) >= 50 and len(self.training_data) % 20 == 0:
                # 定期的に再訓練
                self.train_classifiers()
            
            if self.rf_classifier is not None:
                prediction = self.predict_force_category(features)
            
            # 推奨アクション生成
            if prediction:
                recommendation = self.generate_force_recommendation(data, prediction)
            else:
                # 初期段階では安全な範囲内の力を推奨
                recommendation = {
                    'recommended_force': min(current_force + 1.0, self.safe_force_threshold),
                    'action': '学習中',
                    'safety_level': 'learning',
                    'confidence': 0.0
                }
            
            # 分類結果表示
            category_names = list(self.FORCE_CATEGORIES.keys())
            current_category = category_names[force_category]
            
            print(f"🎯 分類結果: {current_category}")
            print(f"⚖️  現在力: {current_force:.2f}N")
            print(f"💡 推奨アクション: {recommendation['action']}")
            print(f"🔧 推奨力: {recommendation['recommended_force']:.2f}N")
            
            # データをCSVに保存
            self.save_session_data({
                'episode': self.current_episode,
                'step': self.current_step,
                'current_force': current_force,
                'accumulated_force': accumulated_force,
                'is_crushed': is_crushed,
                'predicted_category': current_category,
                'recommended_force': recommendation['recommended_force'],
                'confidence': recommendation['confidence'],
                'timestamp': timestamp
            })
            
            self.current_step += 1
            
            # 応答データ作成
            return {
                "type": "classification_result",
                "force_category": current_category,
                "category_id": force_category,
                "recommended_force": recommendation['recommended_force'],
                "action": recommendation['action'],
                "confidence": recommendation['confidence'],
                "current_step": self.current_step,
                "episode": self.current_episode,
                "timestamp": time.time()
            }
        
        elif message_type == 'episode_end':
            print(f"🏁 エピソード {self.current_episode} 終了")
            response = {
                "type": "episode_ack",
                "total_episodes": self.current_episode + 1,
                "training_samples": len(self.training_data),
                "timestamp": time.time()
            }
            self.current_episode += 1
            self.current_step = 0
            return response
        
        elif message_type == 'reset':
            print("🔄 シミュレーション リセット")
            return {
                "type": "reset_complete",
                "message": "Force Boundary Classifier リセット完了",
                "timestamp": time.time()
            }
        
        # デフォルト応答
        return {
            "type": "ack",
            "message": f"メッセージ受信: {message_type}",
            "timestamp": time.time()
        }

    def run(self, host='127.0.0.1', port=12345):
        """
        分離したTCP通信部分を使ってサーバーを開始
        """
        print(f"🚀 Force Boundary Classifierサーバー開始")
        print(f"📡 接続先: {host}:{port}")
        print(f"🎯 目標: 最適な把持力境界線の学習・分類")
        print("=" * 60)
        
        # 分離したTCP通信サーバーを使用
        server = UnityTcpServer(host, port)
        server.set_message_handler(self.handle_message)
        
        try:
            server.run()
        except KeyboardInterrupt:
            print("\n⏹️ サーバー停止中...")
            server.stop()

if __name__ == "__main__":
    classifier = ForceBoundaryClassifier()
    classifier.run()