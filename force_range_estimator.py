import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
import matplotlib
import platform

def setup_japanese_font():
    system = platform.system()
    try:
        if system == "Windows":
            matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo']
        elif system == "Darwin":
            matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic']
        else:
            matplotlib.rcParams['font.family'] = ['Takao', 'IPAexGothic', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False

JAPANESE_AVAILABLE = setup_japanese_font()

class AIForceBoundaryClassifier:
    """
    AI手法による力境界分類システム
    論文のEEGNetや深層学習手法を力分類に応用
    """
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.data = None
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.feature_importance = None
        
    def load_and_preprocess_data(self):
        """データの読み込みと前処理"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"データを読み込みました: {len(self.data)} 行")
            
            # 特徴量エンジニアリング
            self.data = self.create_features(self.data)
            
            print(f"特徴量数: {len(self.get_feature_columns())}")
            print(f"つぶれた事例: {len(self.data[self.data['is_crushed']])} 件")
            print(f"安全事例: {len(self.data[~self.data['is_crushed']])} 件")
            
            return True
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False
    
    def create_features(self, df):
        """特徴量エンジニアリング - 論文のアプローチに基づく"""
        df = df.copy()
        
        # 基本的な統計的特徴量
        df['force_squared'] = df['current_force'] ** 2
        df['force_log'] = np.log1p(df['current_force'])
        df['force_sqrt'] = np.sqrt(df['current_force'])
        
        # 時系列特徴量（論文の時系列解析に基づく）
        window_sizes = [3, 5, 10]
        for window in window_sizes:
            # 移動平均
            df[f'force_ma_{window}'] = df['current_force'].rolling(window=window, min_periods=1).mean()
            # 移動標準偏差
            df[f'force_std_{window}'] = df['current_force'].rolling(window=window, min_periods=1).std().fillna(0)
            # 力の変化率
            df[f'force_change_{window}'] = df['current_force'].diff(window).fillna(0)
        
        # 累積統計量
        df['force_cumsum'] = df['current_force'].cumsum()
        df['force_cummax'] = df['current_force'].cummax()
        df['force_cummin'] = df['current_force'].cummin()
        
        # 局所的な極値検出
        df['force_local_max'] = df['current_force'].rolling(5, center=True).max() == df['current_force']
        df['force_local_min'] = df['current_force'].rolling(5, center=True).min() == df['current_force']
        
        # 力の安定性指標
        df['force_stability'] = 1.0 / (1.0 + df['force_std_5'])
        
        # ステップ情報を時間的特徴量として使用（存在する場合のみ）
        if 'step' in df.columns:
            max_step = df['step'].max() if df['step'].max() > 0 else 1
            df['step_normalized'] = df['step'] / max_step
        else:
            df['step_normalized'] = 0
        
        return df
    
    def get_feature_columns(self):
        """特徴量列名を取得"""
        exclude_cols = ['episode', 'step', 'is_crushed', 'predicted_category', 
                       'recommended_force', 'confidence', 'timestamp', 'accumulated_force']
        return [col for col in self.data.columns if col not in exclude_cols]
    
    def build_deep_neural_network(self, input_dim):
        """論文のEEGNetにインスパイアされた深層ニューラルネットワーク"""
        model = Sequential([
            # 入力層
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # 隠れ層1（論文の32ユニットを拡張）
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # 隠れ層2
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # 隠れ層3
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            # 出力層（バイナリ分類）
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_all_models(self):
        """複数のAIモデルを訓練"""
        if self.data is None:
            print("先にデータを読み込んでください")
            return
        
        # 特徴量とターゲット準備
        feature_cols = self.get_feature_columns()
        X = self.data[feature_cols].fillna(0).values
        y = self.data['is_crushed'].values.astype(int)
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # 1. Random Forest（アンサンブル学習）
        print("🌲 Random Forest を訓練中...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.models['random_forest'] = rf_grid.best_estimator_
        
        # 2. XGBoost（勾配ブースティング）
        print("🚀 XGBoost を訓練中...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        self.models['xgboost'] = xgb_grid.best_estimator_
        
        # 3. Support Vector Machine
        print("⚡ SVM を訓練中...")
        svm_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        svm = SVC(probability=True, random_state=42)
        svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='roc_auc', n_jobs=-1)
        svm_grid.fit(X_train_scaled, y_train)
        self.models['svm'] = svm_grid.best_estimator_
        
        # 4. 深層ニューラルネットワーク（論文のEEGNetスタイル）
        print("🧠 深層ニューラルネットワーク を訓練中...")
        dnn = self.build_deep_neural_network(X_train_scaled.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        
        history = dnn.fit(
            X_train_scaled, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.models['deep_neural_network'] = dnn
        self.training_history = history
        
        # 5. アンサンブル学習（論文の手法統合）
        print("🎯 アンサンブルモデル を構築中...")
        ensemble_models = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost']),
            ('svm', self.models['svm'])
        ]
        
        # Voting Classifier
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        self.models['ensemble'] = ensemble
        
        # モデル評価
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.X_train = X_train_scaled
        self.y_train = y_train
        
        self.evaluate_all_models()
        self.select_best_model()
        
    def evaluate_all_models(self):
        """全モデルの性能評価"""
        print("\n" + "="*60)
        print("           🏆 AI モデル性能評価")
        print("="*60)
        
        model_scores = {}
        
        for name, model in self.models.items():
            if name == 'deep_neural_network':
                y_pred_proba = model.predict(self.X_test).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                y_pred = model.predict(self.X_test)
            
            # 各種スコア計算
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            accuracy = (y_pred == self.y_test).mean()
            
            # クロスバリデーションスコア
            if name != 'deep_neural_network':
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_mean = cv_std = 0  # DNNのクロスバリデーションは時間がかかるためスキップ
            
            model_scores[name] = {
                'auc': auc_score,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred_proba
            }
            
            print(f"\n📊 {name.upper()}:")
            print(f"  - AUC Score: {auc_score:.4f}")
            print(f"  - Accuracy: {accuracy:.4f}")
            if cv_mean > 0:
                print(f"  - CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
        
        self.model_scores = model_scores
        
    def select_best_model(self):
        """最良モデルの選択"""
        best_auc = 0
        best_name = None
        
        for name, scores in self.model_scores.items():
            if scores['auc'] > best_auc:
                best_auc = scores['auc']
                best_name = name
        
        self.best_model = self.models[best_name]
        print(f"\n🥇 最良モデル: {best_name.upper()} (AUC: {best_auc:.4f})")
        
        # 特徴量重要度を取得
        if hasattr(self.best_model, 'feature_importances_'):
            feature_cols = self.get_feature_columns()
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📈 重要な特徴量 Top 5:")
            for idx, row in self.feature_importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def predict_crush_probability(self, force_value, additional_features=None):
        """指定した力でつぶれる確率を予測（AIモデル使用）"""
        if self.best_model is None:
            print("先にモデルを訓練してください")
            return None
        
        try:
            # 基本特徴量を作成
            temp_df = pd.DataFrame({
                'current_force': [force_value],
                'step': [0],  # デフォルト値を追加
                'accumulated_force': [0]  # デフォルト値を追加
            })
            temp_df = self.create_features(temp_df)
            
            # 追加特徴量があれば統合
            if additional_features:
                for key, value in additional_features.items():
                    temp_df[key] = value
            
            # 特徴量を準備
            feature_cols = self.get_feature_columns()
            
            # 存在しない特徴量は0で埋める
            for col in feature_cols:
                if col not in temp_df.columns:
                    temp_df[col] = 0
            
            X = temp_df[feature_cols].fillna(0).values
            X_scaled = self.scalers['main'].transform(X)
            
            # 予測
            if hasattr(self.best_model, 'predict_proba'):
                prob = self.best_model.predict_proba(X_scaled)[0][1]
            else:  # Deep Neural Network
                prob = self.best_model.predict(X_scaled)[0][0]
            
            return float(prob)
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return 0.5  # デフォルト値を返す
    
    def find_optimal_force_range(self, target_crush_prob=0.05, force_range=None):
        """AI予測に基づく最適力範囲の決定"""
        if force_range is None:
            min_force = self.data['current_force'].min()
            max_force = self.data['current_force'].max()
            force_range = np.linspace(min_force, max_force, 200)
        
        probabilities = [self.predict_crush_probability(f) for f in force_range]
        
        # 目標確率以下の範囲を見つける
        safe_indices = np.where(np.array(probabilities) <= target_crush_prob)[0]
        
        if len(safe_indices) == 0:
            print(f"⚠️  目標確率 {target_crush_prob} 以下の安全な力は見つかりませんでした")
            return None, None
        
        safe_min = force_range[safe_indices[0]]
        safe_max = force_range[safe_indices[-1]]
        
        # 最適範囲（安全範囲の中央50%）
        optimal_min = safe_min + (safe_max - safe_min) * 0.25
        optimal_max = safe_min + (safe_max - safe_min) * 0.75
        
        print(f"\n🎯 AI予測による最適力範囲:")
        print(f"  安全範囲: {safe_min:.3f} - {safe_max:.3f} N (つぶれる確率 ≤ {target_crush_prob})")
        print(f"  最適範囲: {optimal_min:.3f} - {optimal_max:.3f} N")
        
        return (safe_min, safe_max), (optimal_min, optimal_max)
    
    def visualize_ai_results(self):
        """AI分析結果の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        labels = {
            'title': 'AI手法による力境界分析' if JAPANESE_AVAILABLE else 'AI-based Force Boundary Analysis',
            'force': '力 (N)' if JAPANESE_AVAILABLE else 'Force (N)',
            'probability': 'つぶれる確率' if JAPANESE_AVAILABLE else 'Crush Probability',
            'model_comparison': 'モデル性能比較' if JAPANESE_AVAILABLE else 'Model Performance Comparison',
            'roc_curves': 'ROC曲線' if JAPANESE_AVAILABLE else 'ROC Curves',
            'feature_importance': '特徴量重要度' if JAPANESE_AVAILABLE else 'Feature Importance',
            'prediction_surface': '予測面' if JAPANESE_AVAILABLE else 'Prediction Surface',
            'training_history': '訓練履歴' if JAPANESE_AVAILABLE else 'Training History'
        }
        
        # 1. モデル性能比較
        model_names = list(self.model_scores.keys())
        auc_scores = [self.model_scores[name]['auc'] for name in model_names]
        
        axes[0,0].bar(model_names, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink'])
        axes[0,0].set_title(labels['model_comparison'])
        axes[0,0].set_ylabel('AUC Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. ROC曲線
        for name in model_names:
            if name in self.model_scores:
                fpr, tpr, _ = roc_curve(self.y_test, self.model_scores[name]['predictions'])
                auc = self.model_scores[name]['auc']
                axes[0,1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title(labels['roc_curves'])
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 特徴量重要度
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[0,2].barh(range(len(top_features)), top_features['importance'])
            axes[0,2].set_yticks(range(len(top_features)))
            axes[0,2].set_yticklabels(top_features['feature'])
            axes[0,2].set_title(labels['feature_importance'])
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. 力 vs つぶれる確率
        force_range = np.linspace(self.data['current_force'].min(), 
                                self.data['current_force'].max(), 100)
        probabilities = [self.predict_crush_probability(f) for f in force_range]
        
        axes[1,0].plot(force_range, probabilities, 'b-', linewidth=2, label='AI予測')
        axes[1,0].axhline(y=0.05, color='r', linestyle='--', label='5%リスクライン')
        axes[1,0].axhline(y=0.1, color='orange', linestyle='--', label='10%リスクライン')
        axes[1,0].set_xlabel(labels['force'])
        axes[1,0].set_ylabel(labels['probability'])
        axes[1,0].set_title(labels['prediction_surface'])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. 実データとの比較
        safe_data = self.data[self.data['is_crushed'] == False]
        crushed_data = self.data[self.data['is_crushed'] == True]
        
        axes[1,1].scatter(safe_data['current_force'], [0]*len(safe_data), 
                         alpha=0.6, color='green', label='安全', s=20)
        axes[1,1].scatter(crushed_data['current_force'], [1]*len(crushed_data), 
                         alpha=0.6, color='red', label='つぶれた', s=20)
        axes[1,1].plot(force_range, probabilities, 'b-', linewidth=2, label='AI予測曲線')
        axes[1,1].set_xlabel(labels['force'])
        axes[1,1].set_ylabel('実際の結果 / 予測確率')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. 訓練履歴（DNNの場合）
        if hasattr(self, 'training_history'):
            history = self.training_history.history
            axes[1,2].plot(history['loss'], label='Training Loss')
            axes[1,2].plot(history['val_loss'], label='Validation Loss')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Loss')
            axes[1,2].set_title(labels['training_history'])
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_ai_report(self):
        """AI分析結果のレポート生成"""
        print("\n" + "="*60)
        print("           🤖 AI手法による力境界分析レポート")
        print("="*60)
        
        # データ概要
        print(f"\n📊 データ概要:")
        print(f"  - 総データ数: {len(self.data)}")
        print(f"  - 特徴量数: {len(self.get_feature_columns())}")
        print(f"  - つぶれた事例: {len(self.data[self.data['is_crushed']])} 件")
        print(f"  - 安全事例: {len(self.data[~self.data['is_crushed']])} 件")
        
        # 最良モデルの性能
        if self.best_model is not None:
            best_name = None
            for name, model in self.models.items():
                if model == self.best_model:
                    best_name = name
                    break
            
            print(f"\n🥇 最良モデル: {best_name.upper()}")
            print(f"  - AUC Score: {self.model_scores[best_name]['auc']:.4f}")
            print(f"  - Accuracy: {self.model_scores[best_name]['accuracy']:.4f}")
        
        # AI予測による最適範囲
        safe_range, optimal_range = self.find_optimal_force_range()
        
        # リスク評価例
        print(f"\n⚠️  リスク評価例:")
        test_forces = np.linspace(self.data['current_force'].min(), 
                                self.data['current_force'].max(), 5)
        for force in test_forces:
            prob = self.predict_crush_probability(force)
            risk_level = "低" if prob < 0.05 else "中" if prob < 0.2 else "高"
            print(f"  - {force:.2f} N: つぶれる確率 {prob*100:.1f}% (リスク: {risk_level})")
        
        print("\n" + "="*60)


# 使用例
if __name__ == "__main__":
    # CSVファイルのパス
    csv_file = "force_boundary_classification_20250812_002330.csv"
    
    # AI分類器を初期化
    ai_classifier = AIForceBoundaryClassifier(csv_file)
    
    # データを読み込み・前処理
    if ai_classifier.load_and_preprocess_data():
        print("✅ データの読み込み完了")
        
        # 全AIモデルを訓練
        print("\n🚀 AI訓練開始...")
        ai_classifier.train_all_models()
        
        # 結果可視化
        ai_classifier.visualize_ai_results()
        
        # レポート生成
        ai_classifier.generate_ai_report()
        
        # 個別予測テスト
        test_force = 20.0
        prob = ai_classifier.predict_crush_probability(test_force)
        print(f"\n🧪 テスト: {test_force} Nでつぶれる確率は {prob*100:.1f}%です")