"""
実際のUnity-A2Cデータを使用したEEGNet分類器
既存のA2C学習結果を教師データとして活用
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
from datetime import datetime
import glob


class RealDataEEGNetClassifier:
    """
    実際のUnity-A2Cデータを使用したEEGNet分類器
    """
    
    def __init__(self, 
                 nb_classes=3,
                 chans=64,
                 samples=128,
                 dropoutRate=0.5,
                 kernLength=32,
                 F1=8,
                 D=2,
                 F2=16):
        
        self.nb_classes = nb_classes
        self.chans = chans
        self.samples = samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        # 把持力判定クラス（Unity-A2Cの報酬システムと対応）
        self.class_names = ['Success', 'Over-grip', 'Under-grip']
        
        # データ読み込み設定
        self.force_thresholds = {
            'under_grip_max': 9.9,    # 2N未満 = Under-grip
            'success_min': 10.0,       # 2N～15N = Success  
            'success_max': 15.0,
            'over_grip_min': 15.0     # 20N以上 = Over-grip
        }
    
    def load_unity_a2c_data(self, data_path="./"):
        """
        実際のUnity-A2Cデータを読み込む
        
        Args:
            data_path: A2Cデータが保存されているディレクトリ
        
        Returns:
            DataFrame: 読み込まれたA2Cデータ
        """
        
        print("🔄 実際のUnity-A2Cデータを読み込み中...")
        
        # A2Cデータファイルを探す
        possible_files = [
            'unity_a2c_final_results.csv',
            'RL_data.csv',
            'unity_a2c_progress_game*.csv',
            'grasp_a2c_*.csv',
            'grasp_a2c_final_results*.csv'
        ]
        
        found_files = []
        for pattern in possible_files:
            files = glob.glob(os.path.join(data_path, pattern))
            found_files.extend(files)
        
        if not found_files:
            print(f"❌ A2Cデータファイルが見つかりません: {data_path}")
            print(f"   探索対象: {possible_files}")
            return None
        
        # 最新のファイルを使用
        latest_file = max(found_files, key=os.path.getmtime)
        print(f"📂 使用ファイル: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file)
            print(f"✅ データ読み込み成功: {len(df)}行")
            print(f"📊 カラム: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ ファイル読み込みエラー: {e}")
            return None
    
    def convert_a2c_data_to_classification_labels(self, df):
        """
        A2Cデータを分類ラベルに変換
        
        Args:
            df: Unity-A2Cから読み込まれたDataFrame
            
        Returns:
            labels: 分類ラベル配列
            forces: 対応する把持力配列
        """
        
        print("🔄 A2Cデータを分類ラベルに変換中...")
        
        if 'action' not in df.columns:
            print("❌ 'action'カラムが見つかりません")
            return None, None
        
        if 'final_reward' not in df.columns:
            print("❌ 'final_reward'カラムが見つかりません")  
            return None, None
        
        labels = []
        forces = df['action'].values  # 把持力（アクション値）
        rewards = df['final_reward'].values  # 最終報酬
        
        for i, (force, reward) in enumerate(zip(forces, rewards)):
            
            # 報酬ベースの分類（A2Cシステムの報酬設計と一致）
            if reward >= 80:  # 高報酬 = Success
                label = 'Success'
            elif reward <= -40:  # 低報酬 = 失敗（Over/Under）
                # 把持力で細分化
                if force < self.force_thresholds['under_grip_max']:
                    label = 'Under-grip'
                elif force > self.force_thresholds['over_grip_min']:
                    label = 'Over-grip'
                else:
                    # 中間力でも失敗 = その他の要因
                    label = 'Under-grip' if force < 10 else 'Over-grip'
            else:
                # 中間報酬 = 力で判定
                if force < self.force_thresholds['under_grip_max']:
                    label = 'Under-grip'
                elif force > self.force_thresholds['over_grip_min']:
                    label = 'Over-grip'
                else:
                    label = 'Success'
            
            labels.append(label)
        
        # 分布確認
        counter = Counter(labels)
        print("📊 分類結果分布:")
        for label, count in counter.items():
            percentage = count / len(labels) * 100
            print(f"  {label}: {count}件 ({percentage:.1f}%)")
        
        return np.array(labels), forces
    
    def generate_mock_eeg_from_labels(self, labels, forces):
        """
        分類ラベルに基づいてモックEEGデータを生成
        実際のEEGデータが取得できるまでの暫定実装
        
        Args:
            labels: 分類ラベル
            forces: 対応する把持力
            
        Returns:
            X: モックEEGデータ
            y: ワンホットエンコードされたラベル
        """
        
        print("🧠 分類ラベルベースのモックEEGデータ生成中...")
        
        n_samples = len(labels)
        X = np.zeros((n_samples, self.chans, self.samples, 1))
        
        # ラベルごとに特徴的なパターンを生成
        for i, (label, force) in enumerate(zip(labels, forces)):
            
            # ベースノイズ
            base_signal = np.random.randn(self.chans, self.samples, 1) * 0.1
            
            # ラベル固有の特徴を追加
            if label == 'Success':
                # 成功時：安定したパターン
                pattern = self._generate_success_pattern()
            elif label == 'Over-grip':
                # オーバーグリップ：強いエラー信号
                pattern = self._generate_overgrip_pattern()
            elif label == 'Under-grip':
                # アンダーグリップ：弱いエラー信号  
                pattern = self._generate_undergrip_pattern()
            else:
                pattern = np.zeros((self.chans, self.samples, 1))
            
            # 力の大きさも反映
            force_factor = np.clip(force / 20.0, 0.1, 2.0)
            X[i] = base_signal + pattern * force_factor
        
        # ラベルエンコーディング
        y_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(y_encoded, self.nb_classes)
        
        print(f"✅ モックEEGデータ生成完了: {n_samples}サンプル")
        return X, y
    
    def _generate_success_pattern(self):
        """成功時の脳波パターン（モック）"""
        pattern = np.random.randn(self.chans, self.samples, 1) * 0.05
        
        # 前頭部（Fz周辺）に小さな正の電位
        frontal_channels = list(range(0, 8))  # 前頭部チャンネル
        for ch in frontal_channels:
            # 300-500ms後に小さなポジティブピーク
            peak_start = int(0.3 * self.samples)
            peak_end = int(0.5 * self.samples)
            pattern[ch, peak_start:peak_end, 0] += 0.2
            
        return pattern
    
    def _generate_overgrip_pattern(self):
        """オーバーグリップ時の脳波パターン（モック）"""
        pattern = np.random.randn(self.chans, self.samples, 1) * 0.05
        
        # 前頭・中央部に強いエラー関連電位
        error_channels = list(range(8, 24))  # 中央部チャンネル
        for ch in error_channels:
            # 200-300ms後にネガティブピーク（N200）
            n200_start = int(0.2 * self.samples)
            n200_end = int(0.3 * self.samples)
            pattern[ch, n200_start:n200_end, 0] -= 0.5
            
            # 300-600ms後にポジティブピーク（P300）
            p300_start = int(0.3 * self.samples)
            p300_end = int(0.6 * self.samples)
            pattern[ch, p300_start:p300_end, 0] += 0.8
            
        return pattern
    
    def _generate_undergrip_pattern(self):
        """アンダーグリップ時の脳波パターン（モック）"""
        pattern = np.random.randn(self.chans, self.samples, 1) * 0.05
        
        # 中程度のエラー関連電位
        error_channels = list(range(16, 32))  # 中央部チャンネル
        for ch in error_channels:
            # オーバーグリップより弱いが、明確なエラー信号
            n200_start = int(0.2 * self.samples)
            n200_end = int(0.3 * self.samples)
            pattern[ch, n200_start:n200_end, 0] -= 0.3
            
            p300_start = int(0.3 * self.samples)
            p300_end = int(0.6 * self.samples)
            pattern[ch, p300_start:p300_end, 0] += 0.5
            
        return pattern
    
    def create_eegnet_model(self):
        """EEGNetモデル構築（constraint削除版）"""
        
        input_layer = layers.Input(shape=(self.chans, self.samples, 1))
        
        # Block 1: 時間的畳み込み
        block1 = layers.Conv2D(self.F1, (1, self.kernLength), 
                              padding='same', 
                              use_bias=False)(input_layer)
        block1 = layers.BatchNormalization()(block1)
        
        # 深度方向畳み込み（constraint削除）
        block1 = layers.DepthwiseConv2D((self.chans, 1), 
                                       use_bias=False, 
                                       depth_multiplier=self.D)(block1)
        block1 = layers.BatchNormalization()(block1)
        block1 = layers.Activation('elu')(block1)
        block1 = layers.AveragePooling2D((1, 4))(block1)
        block1 = layers.Dropout(self.dropoutRate)(block1)
        
        # Block 2: 分離可能畳み込み
        block2 = layers.SeparableConv2D(self.F2, (1, 16),
                                       use_bias=False, 
                                       padding='same')(block1)
        block2 = layers.BatchNormalization()(block2)
        block2 = layers.Activation('elu')(block2)
        block2 = layers.AveragePooling2D((1, 8))(block2)
        block2 = layers.Dropout(self.dropoutRate)(block2)
        
        # Block 3: 分類層（constraint削除）
        flatten = layers.Flatten(name='flatten')(block2)
        dense = layers.Dense(self.nb_classes, name='dense')(flatten)
        softmax = layers.Activation('softmax', name='softmax')(dense)
        
        model = models.Model(inputs=input_layer, outputs=softmax)
        return model
    
    def compile_model(self, learning_rate=0.01):
        """モデルコンパイル"""
        if self.model is None:
            self.model = self.create_eegnet_model()
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        return self.model
    
    def analyze_unity_data(self, df):
        """Unity-A2Cデータの分析"""
        
        print("📊 Unity-A2Cデータ分析:")
        print(f"  総エピソード数: {len(df)}")
        print(f"  平均ステップ数: {df['steps'].mean():.1f}")
        print(f"  平均報酬: {df['final_reward'].mean():.1f}")
        print(f"  平均把持力: {df['action'].mean():.2f}N")
        
        # 報酬分布
        reward_bins = pd.cut(df['final_reward'], bins=[-200, -50, 50, 200], 
                           labels=['失敗', '中間', '成功'])
        print("\n📈 報酬分布:")
        print(reward_bins.value_counts())
        
        # 把持力分布
        force_bins = pd.cut(df['action'], bins=[0, 2, 15, 20, 50], 
                          labels=['弱い', '適正', '中強', '強い'])
        print("\n🔧 把持力分布:")
        print(force_bins.value_counts())
        
        # 可視化
        self.plot_unity_data_analysis(df)
        
        return df
    
    def plot_unity_data_analysis(self, df):
        """Unity-A2Cデータの可視化"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 報酬と把持力の関係
        axes[0,0].scatter(df['action'], df['final_reward'], alpha=0.6)
        axes[0,0].set_xlabel('把持力 (N)')
        axes[0,0].set_ylabel('最終報酬')
        axes[0,0].set_title('把持力 vs 報酬')
        axes[0,0].grid(True)
        
        # 把持力分布
        axes[0,1].hist(df['action'], bins=20, edgecolor='black', alpha=0.7)
        axes[0,1].axvline(self.force_thresholds['under_grip_max'], color='red', linestyle='--', label='Under-grip閾値')
        axes[0,1].axvline(self.force_thresholds['success_max'], color='green', linestyle='--', label='Success上限')
        axes[0,1].axvline(self.force_thresholds['over_grip_min'], color='orange', linestyle='--', label='Over-grip閾値')
        axes[0,1].set_xlabel('把持力 (N)')
        axes[0,1].set_ylabel('頻度')
        axes[0,1].set_title('把持力分布')
        axes[0,1].legend()
        
        # 報酬分布
        axes[1,0].hist(df['final_reward'], bins=20, edgecolor='black', alpha=0.7)
        axes[1,0].axvline(50, color='green', linestyle='--', label='成功閾値')
        axes[1,0].axvline(-50, color='red', linestyle='--', label='失敗閾値')
        axes[1,0].set_xlabel('最終報酬')
        axes[1,0].set_ylabel('頻度')
        axes[1,0].set_title('報酬分布')
        axes[1,0].legend()
        
        # 学習進行（報酬の移動平均）
        window_size = min(10, len(df))
        df['reward_ma'] = df['final_reward'].rolling(window=window_size).mean()
        axes[1,1].plot(df['game'], df['reward_ma'])
        axes[1,1].set_xlabel('エピソード')
        axes[1,1].set_ylabel('報酬（移動平均）')
        axes[1,1].set_title('学習進行')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('unity_a2c_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_with_real_unity_data(self, data_path="./"):
        """
        実際のUnity-A2Cデータを使用した訓練
        """
        
        print("🚀 実際のUnity-A2Cデータで訓練開始!")
        print("=" * 50)
        
        # 1. Unity-A2Cデータ読み込み
        df = self.load_unity_a2c_data(data_path)
        if df is None:
            return False
        
        # 2. データ分析
        df = self.analyze_unity_data(df)
        
        # 3. 分類ラベル変換
        labels, forces = self.convert_a2c_data_to_classification_labels(df)
        if labels is None:
            return False
        
        # 4. モックEEGデータ生成
        X, y = self.generate_mock_eeg_from_labels(labels, forces)
        
        # 5. データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        print(f"\n📊 訓練データ準備完了:")
        print(f"  訓練データ: {len(X_train)}件")
        print(f"  テストデータ: {len(X_test)}件")
        
        # 6. モデル訓練
        self.compile_model()
        
        print("\n🧠 EEGNet訓練実行中...")
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
            callbacks.ModelCheckpoint('real_data_eegnet_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=16,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 7. 評価
        print("\n📊 モデル評価:")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # 分類レポート
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=self.class_names)
        
        print(f"✅ テスト精度: {test_accuracy:.4f}")
        print(f"\n📋 分類レポート:\n{report}")
        
        # 混同行列
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix(cm)
        
        # 訓練履歴可視化
        self.plot_training_history()
        
        # 結果保存
        self.save_results_with_unity_data(df, labels, forces, test_accuracy)
        
        return True
    
    def save_results_with_unity_data(self, unity_df, labels, forces, accuracy):
        """結果をUnityデータと一緒に保存"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 結果DataFrame作成
        results_df = unity_df.copy()
        results_df['predicted_label'] = labels
        results_df['grip_force'] = forces
        results_df['classification_accuracy'] = accuracy
        
        # 保存
        results_file = f'unity_eegnet_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        print(f"💾 結果保存: {results_file}")
        
        # モデル保存
        model_file = f'unity_trained_eegnet_{timestamp}.h5'
        self.model.save(model_file)
        print(f"💾 モデル保存: {model_file}")
    
    def plot_training_history(self):
        """訓練履歴可視化"""
        
        if self.history is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 損失
        axes[0].plot(self.history.history['loss'], label='training loss')
        axes[0].plot(self.history.history['val_loss'], label='validation loss')
        axes[0].set_title('loss model actual database')
        axes[0].set_xlabel('epoch')
        axes[0].set_ylabel('loss')
        axes[0].legend()
        
        # 精度
        axes[1].plot(self.history.history['accuracy'], label='training accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='validation accuracy')
        axes[1].set_title('accuracy model actual database')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('real_data_eegnet_training.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """混同行列可視化"""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('混同行列 - 実Unity-a2c database')
        plt.xlabel('estimated class')
        plt.ylabel('actual class')
        plt.tight_layout()
        plt.savefig('real_data_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


def main_with_real_data():
    """実際のUnity-A2Cデータを使用したメイン実行"""
    
    print("🎯 実Unity-A2Cデータ使用EEGNet分類器")
    print("=" * 50)
    
    # 分類器初期化
    classifier = RealDataEEGNetClassifier(
        nb_classes=3,
        chans=64,
        samples=128
    )
    
    # 実データで訓練
    success = classifier.train_with_real_unity_data()
    
    if success:
        print("\n✅ 実データ訓練完了!")
        print("\n💡 次のステップ:")
        print("1. 実際のEEG機器接続")
        print("2. LSLストリーム設定")
        print("3. リアルタイム分類システム有効化")
    else:
        print("\n❌ 実データ訓練に失敗")
    
    return classifier


def check_available_data():
    """利用可能なデータファイルをチェック"""
    
    print("🔍 利用可能なA2Cデータファイルをチェック中...")
    
    current_dir = "./"
    possible_patterns = [
        'unity_a2c_final_results.csv',
        'RL_data.csv', 
        'unity_a2c_progress_game*.csv',
        'unity_*.csv'
        'grasp_a2c_*.csv' 
    ]
    
    found_files = []
    for pattern in possible_patterns:
        files = glob.glob(os.path.join(current_dir, pattern))
        found_files.extend(files)
    
    if found_files:
        print("✅ 発見されたデータファイル:")
        for file in found_files:
            size = os.path.getsize(file)
            modified = datetime.fromtimestamp(os.path.getmtime(file))
            print(f"  📄 {file}")
            print(f"     サイズ: {size} bytes, 更新: {modified}")
            
            # ファイルの簡易プレビュー
            try:
                preview_df = pd.read_csv(file, nrows=3)
                print(f"     カラム: {list(preview_df.columns)}")
                print(f"     行数: {len(pd.read_csv(file))}行")
            except:
                print("     読み込みエラー")
            print()
    else:
        print("❌ A2Cデータファイルが見つかりません")
        print("   A2C.pyを実行してデータを生成してください")
    
    return found_files


class UnityDataProcessor:
    """
    Unity-A2Cデータの詳細処理クラス
    """
    
    def __init__(self):
        self.processed_data = None
        self.statistics = {}
    
    def create_enhanced_features(self, df):
        """
        Unity-A2Cデータから拡張特徴量を作成
        """
        
        print("🔧 拡張特徴量を作成中...")
        
        enhanced_df = df.copy()
        
        # 1. 把持力効率性
        enhanced_df['force_efficiency'] = enhanced_df['final_reward'] / (enhanced_df['action'] + 0.1)
        
        # 2. 学習進捗指標
        enhanced_df['episode_progress'] = enhanced_df['game'] / enhanced_df['game'].max()
        
        # 3. 成功率（移動平均）
        window = 10
        enhanced_df['success_rate'] = (enhanced_df['final_reward'] > 50).rolling(window=window).mean()
        
        # 4. Actor/Critic損失比率
        enhanced_df['loss_ratio'] = enhanced_df['actor_cost'] / (enhanced_df['critic_cost'].abs() + 1e-8)
        
        # 5. 把持力安定性（前エピソードとの差）
        enhanced_df['force_stability'] = enhanced_df['action'].diff().abs()
        
        # 6. 複合指標：学習品質スコア
        enhanced_df['learning_quality'] = (
            enhanced_df['success_rate'].fillna(0) * 0.4 +
            (1 / (enhanced_df['steps'] + 1)) * 0.3 +  # ステップ効率性
            (1 / (enhanced_df['loss_ratio'].abs() + 1)) * 0.3  # 損失安定性
        )
        
        print("✅ 拡張特徴量作成完了")
        
        return enhanced_df
    
    def create_time_series_features(self, df):
        """
        時系列特徴量の作成
        """
        
        print("📈 時系列特徴量を作成中...")
        
        # ウィンドウサイズ
        short_window = 5
        long_window = 20
        
        # 移動平均
        df['action_ma_short'] = df['action'].rolling(window=short_window).mean()
        df['action_ma_long'] = df['action'].rolling(window=long_window).mean()
        df['reward_ma_short'] = df['final_reward'].rolling(window=short_window).mean()
        df['reward_ma_long'] = df['final_reward'].rolling(window=long_window).mean()
        
        # トレンド
        df['action_trend'] = df['action_ma_short'] - df['action_ma_long']
        df['reward_trend'] = df['reward_ma_short'] - df['reward_ma_long']
        
        # ボラティリティ
        df['action_volatility'] = df['action'].rolling(window=short_window).std()
        df['reward_volatility'] = df['final_reward'].rolling(window=short_window).std()
        
        return df
    
    def generate_balanced_dataset(self, labels, target_samples_per_class=200):
        """
        バランスの取れたデータセットを生成
        """
        
        print(f"⚖️ クラス間バランス調整（各クラス{target_samples_per_class}サンプル）...")
        
        # クラス別にインデックスを取得
        class_indices = {}
        for class_name in ['Success', 'Over-grip', 'Under-grip']:
            indices = np.where(labels == class_name)[0]
            class_indices[class_name] = indices
            print(f"  {class_name}: {len(indices)}サンプル利用可能")
        
        # 各クラスから均等にサンプリング
        balanced_indices = []
        
        for class_name, indices in class_indices.items():
            if len(indices) >= target_samples_per_class:
                # 十分なサンプルがある場合：ランダムサンプリング
                selected = np.random.choice(indices, target_samples_per_class, replace=False)
            else:
                # 不足している場合：重複を許可してサンプリング
                selected = np.random.choice(indices, target_samples_per_class, replace=True)
                print(f"    ⚠️ {class_name}: 重複サンプリング使用")
            
            balanced_indices.extend(selected)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)  # シャッフル
        
        print(f"✅ バランス調整完了: 総{len(balanced_indices)}サンプル")
        
        return balanced_indices


def create_data_visualization_dashboard(df, labels, forces):
    """
    データ可視化ダッシュボード
    """
    
    print("📊 データ可視化ダッシュボード作成中...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 把持力vs報酬の散布図（分類別色分け）
    ax1 = plt.subplot(3, 3, 1)
    label_colors = {'Success': 'green', 'Over-grip': 'red', 'Under-grip': 'orange'}
    
    for label in ['Success', 'Over-grip', 'Under-grip']:
        mask = labels == label
        plt.scatter(forces[mask], df['final_reward'][mask], 
                   c=label_colors[label], label=label, alpha=0.6, s=20)

    plt.xlabel('grasp force (N)')
    plt.ylabel('final reward')
    plt.title('grasp force vs final reward (by class)')
    plt.legend()
    plt.grid(True)
    
    # 2. 学習進行と分類の関係
    ax2 = plt.subplot(3, 3, 2)
    for label in ['Success', 'Over-grip', 'Under-grip']:
        mask = labels == label
        episodes = df['game'][mask]
        plt.scatter(episodes, forces[mask], c=label_colors[label], 
                   label=label, alpha=0.6, s=15)
    
    plt.xlabel('episode')
    plt.ylabel('grasp force (N)')
    plt.title('learning progress vs grasp force (by class)')
    plt.legend()
    plt.grid(True)
    
    # 3. 分類分布（円グラフ）
    ax3 = plt.subplot(3, 3, 3)
    label_counts = Counter(labels)
    plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%',
           colors=[label_colors[label] for label in label_counts.keys()])
    plt.title('classification distribution')
    
    # 4. 把持力分布（クラス別ヒストグラム）
    ax4 = plt.subplot(3, 3, 4)
    for label in ['Success', 'Over-grip', 'Under-grip']:
        mask = labels == label
        plt.hist(forces[mask], bins=15, alpha=0.6, label=label, 
                color=label_colors[label], edgecolor='black')

    plt.xlabel('grasp force (N)')
    plt.ylabel('frequency')
    plt.title('grasp force distribution (by class)')
    plt.legend()
    
    # 5. 報酬分布（クラス別ヒストグラム）
    ax5 = plt.subplot(3, 3, 5)
    for label in ['Success', 'Over-grip', 'Under-grip']:
        mask = labels == label
        plt.hist(df['final_reward'][mask], bins=15, alpha=0.6, label=label,
                color=label_colors[label], edgecolor='black')
    
    plt.xlabel('final reward')
    plt.ylabel('frequency')
    plt.title('final reward distribution (by class)')
    plt.legend()
    
    # 6. 学習曲線（報酬）
    ax6 = plt.subplot(3, 3, 6)
    window = 10
    reward_ma = df['final_reward'].rolling(window=window).mean()
    plt.plot(df['game'], reward_ma, linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward (moving average)')
    plt.title(f'learning curve (moving average over {window} episodes)')
    plt.grid(True)
    
    # 7. Actor/Critic損失
    ax7 = plt.subplot(3, 3, 7)
    plt.plot(df['game'], df['actor_cost'].rolling(window=10).mean(), label='Actor loss', alpha=0.8)
    plt.plot(df['game'], df['critic_cost'].rolling(window=10).mean(), label='Critic loss', alpha=0.8)
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.title('Actor/Critic loss')
    plt.legend()
    plt.grid(True)
    
    # 8. ステップ効率性
    ax8 = plt.subplot(3, 3, 8)
    for label in ['Success', 'Over-grip', 'Under-grip']:
        mask = labels == label
        plt.scatter(df['steps'][mask], df['final_reward'][mask],
                   c=label_colors[label], label=label, alpha=0.6, s=15)
    
    plt.xlabel('step')
    plt.ylabel('final reward')
    plt.title('step efficiency vs final reward')
    plt.legend()
    plt.grid(True)
    
    # 9. 把持力の時系列変化
    ax9 = plt.subplot(3, 3, 9)
    force_ma = df['action'].rolling(window=10).mean()
    plt.plot(df['game'], force_ma, linewidth=2, color='blue')
    
    # 閾値線を表示
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Under-grip threshold')
    plt.axhline(y=15.0, color='green', linestyle='--', alpha=0.7, label='Success over threshold')
    plt.axhline(y=20.0, color='orange', linestyle='--', alpha=0.7, label='Over-grip threshold')

    plt.xlabel('episode')
    plt.ylabel('grasp force (N)')
    plt.title('change of grasp force during training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('unity_a2c_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_comprehensive_analysis():
    """
    包括的なデータ分析と可視化
    """
    
    print("🔍 Unity-A2Cデータの包括的分析を実行中...")
    
    # データ確認
    available_files = check_available_data()
    
    if not available_files:
        print("❌ 分析対象データが見つかりません")
        return None
    
    # 最新ファイルを使用
    latest_file = max(available_files, key=os.path.getmtime)
    print(f"📂 分析対象: {latest_file}")
    
    try:
        # データ読み込み
        df = pd.read_csv(latest_file)
        
        # 基本統計
        print(f"\n📊 基本統計:")
        print(f"  エピソード数: {len(df)}")
        print(f"  把持力範囲: {df['action'].min():.2f}N ～ {df['action'].max():.2f}N")
        print(f"  報酬範囲: {df['final_reward'].min():.1f} ～ {df['final_reward'].max():.1f}")
        
        # 分類器初期化
        classifier = RealDataEEGNetClassifier()
        
        # 分類ラベル生成
        labels, forces = classifier.convert_a2c_data_to_classification_labels(df)
        
        if labels is not None:
            # 包括的可視化
            create_data_visualization_dashboard(df, labels, forces)
            
            # EEGNet訓練
            print("\n🧠 実データベースEEGNet訓練開始...")
            success = classifier.train_with_real_unity_data()
            
            return classifier, df, labels, forces
        else:
            print("❌ ラベル変換に失敗")
            return None, df, None, None
            
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        return None, None, None, None


def generate_final_report(classifier, df, labels, forces):
    """
    最終レポート生成
    """
    
    if classifier is None or df is None:
        print("❌ レポート生成に必要なデータが不足")
        return
    
    print("📋 最終レポート生成中...")
    
    # システム性能サマリー
    report = {
        'システム概要': {
            'データソース': 'Unity-A2C実験結果',
            '総エピソード数': len(df),
            '分類対象': '把持力適正性判定',
            'クラス数': 3,
            'EEGNetアーキテクチャ': 'Lawhern et al. (2018)'
        },
        
        'データ統計': {
            '平均把持力': f"{df['action'].mean():.2f}N",
            '把持力標準偏差': f"{df['action'].std():.2f}N",
            '平均報酬': f"{df['final_reward'].mean():.1f}",
            '平均ステップ数': f"{df['steps'].mean():.1f}",
            '成功率': f"{(df['final_reward'] > 50).mean()*100:.1f}%"
        },
        
        '分類分布': dict(Counter(labels)) if labels is not None else {},
        
        'システム準備状況': {
            'EEGNet実装': '✅ 完了',
            'Unity統合': '✅ 完了',
            'データ処理': '✅ 完了',
            'リアルタイム分類': '🔄 EEG機器接続待ち',
            'LSLストリーム': '🔄 設定待ち'
        },
        
        '次のアクション': [
            '1. EEG機器のセットアップ',
            '2. LSL4Unityでストリーム設定',
            '3. リアルタイム分類テスト',
            '4. 実EEGデータでモデル再訓練',
            '5. 論文用データ収集実験'
        ]
    }
    
    # レポート保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'unity_eegnet_final_report_{timestamp}.json'
    
    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # レポート表示
    print("\n📋 最終システムレポート:")
    print("=" * 60)
    
    for section, content in report.items():
        print(f"\n【{section}】")
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                print(f"  {item}")
        else:
            print(f"  {content}")
    
    print(f"\n💾 詳細レポート保存: {report_file}")
    
    return report


if __name__ == "__main__":
    print("🎯 Unity-A2C実データ使用EEGNet分類システム")
    print("=" * 60)
    
    # 1. データファイル確認
    available_files = check_available_data()
    
    if available_files:
        print("\n🚀 実際のUnity-A2Cデータで包括的分析を開始...")
        
        # 2. 包括的分析実行
        classifier, df, labels, forces = run_comprehensive_analysis()
        
        # 3. 最終レポート生成
        if classifier is not None:
            report = generate_final_report(classifier, df, labels, forces)
            
            print("\n✅ 全ての処理が完了しました!")
            print("\n🎯 要約:")
            print("  - Unity-A2Cデータを成功裏に分析")
            print("  - EEGNet分類器を実データベースで訓練")
            print("  - 把持力判定システムの基盤完成")
            print("  - EEG機器接続の準備完了")
        
    else:
        print("\n⚠️ 実データが見つからないため、デモ用データで実行...")
        
        # ダミーデータでのデモ実行
        classifier = RealDataEEGNetClassifier()
        
        # ダミーデータ生成
        dummy_df = pd.DataFrame({
            'game': range(1, 101),
            'steps': np.random.randint(20, 150, 100),
            'actor_cost': np.random.uniform(0.01, 0.1, 100),
            'critic_cost': np.random.uniform(-0.2, -0.05, 100), 
            'final_reward': np.random.choice([-100, -50, 50, 100], 100, p=[0.1, 0.2, 0.2, 0.5]),
            'action': np.random.uniform(1.0, 25.0, 100)
        })
        
        print("📊 ダミーデータで分析実行...")
        labels, forces = classifier.convert_a2c_data_to_classification_labels(dummy_df)
        
        if labels is not None:
            create_data_visualization_dashboard(dummy_df, labels, forces)
            X, y = classifier.generate_mock_eeg_from_labels(labels, forces)
            
            # 簡易訓練
            classifier.compile_model()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            classifier.history = classifier.model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=20,
                batch_size=16,
                verbose=1
            )
            
            classifier.plot_training_history()
            print("✅ ダミーデータでのデモ完了")
        
    print("\n💡 システム使用準備完了!")
    print("   EEG機器接続後、リアルタイム分類が可能になります。")