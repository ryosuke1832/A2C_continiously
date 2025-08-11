import numpy as np
import time
import threading
from pylsl import StreamInfo, StreamOutlet
import argparse
import pandas as pd
from datetime import datetime
import os

class EEGSimulator:
    def __init__(self, channels=16, sample_rate=250, stream_name="EEGSimulation_16ch"):
        """
        EEGシミュレーター
        
        Args:
            channels: チャンネル数 (default: 16)
            sample_rate: サンプリング周波数 (default: 250Hz)
            stream_name: LSLストリーム名
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.stream_name = stream_name
        
        # LSLストリーム情報を作成
        info = StreamInfo(
            name=stream_name,
            type='EEG',
            channel_count=channels,
            nominal_srate=sample_rate,
            channel_format='float32',
            source_id='eeg_simulator_001'
        )
        
        # チャンネル情報を設定
        channels_info = info.desc().append_child("channels")
        channel_names = [
            'Fz', 'F3', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 
            'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'Oz'
        ][:channels]
        
        for i, name in enumerate(channel_names):
            ch = channels_info.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
        
        # アウトレット作成
        self.outlet = StreamOutlet(info)
        
        # シミュレーション状態
        self.running = False
        self.thread = None
        self.time_start = time.time()
        
        # 信号生成パラメータ
        self.noise_level = 10.0  # ベースノイズレベル (μV)
        self.alpha_amplitude = 15.0  # アルファ波振幅 (μV)
        self.beta_amplitude = 8.0   # ベータ波振幅 (μV)
        self.theta_amplitude = 12.0  # シータ波振幅 (μV)
        
        # エラー関連電位パラメータ
        self.errp_active = False
        self.errp_start_time = 0
        self.errp_amplitude = 25.0  # ErrP振幅 (μV)
        
        # データ記録用
        self.recording_data = []
        self.errp_events = []
        self.start_time = None
        
    def generate_eeg_sample(self, t):
        """
        1サンプルのEEG様データを生成
        
        Args:
            t: 現在の時刻 (秒)
            
        Returns:
            numpy.array: 各チャンネルの値 (μV)
        """
        sample = np.zeros(self.channels)
        
        for ch in range(self.channels):
            # チャンネル固有の位相オフセット
            phase_offset = ch * np.pi / 8
            
            # 基本的な脳波成分
            # アルファ波 (8-12Hz, 主に10Hz)
            alpha = self.alpha_amplitude * np.sin(2 * np.pi * 10 * t + phase_offset)
            
            # ベータ波 (13-30Hz, 主に20Hz)
            beta = self.beta_amplitude * np.sin(2 * np.pi * 20 * t + phase_offset * 1.5)
            
            # シータ波 (4-8Hz, 主に6Hz)
            theta = self.theta_amplitude * np.sin(2 * np.pi * 6 * t + phase_offset * 0.7)
            
            # ガンマ波 (30-100Hz, 主に40Hz) - 小さな振幅
            gamma = 3.0 * np.sin(2 * np.pi * 40 * t + phase_offset * 2)
            
            # ランダムノイズ
            noise = np.random.normal(0, self.noise_level)
            
            # 基本信号の合成
            signal = alpha + beta + theta + gamma + noise
            
            # エラー関連電位 (ErrP) の追加
            if self.errp_active:
                errp_signal = self._generate_errp(t, ch)
                signal += errp_signal
            
            sample[ch] = signal
            
        return sample
    
    def _generate_errp(self, t, channel):
        """
        エラー関連電位 (ErrP) を生成
        
        Args:
            t: 現在の時刻
            channel: チャンネル番号
            
        Returns:
            float: ErrP成分の値
        """
        if not self.errp_active:
            return 0.0
            
        errp_time = t - self.errp_start_time
        
        # ErrPは通常600ms程度続く
        if errp_time > 0.6:
            self.errp_active = False
            return 0.0
            
        # 前頭部チャンネル (Fz, F3, F4) で強く現れる
        frontal_channels = [0, 1, 2]  # Fz, F3, F4
        amplitude_factor = 1.0 if channel in frontal_channels else 0.3
        
        # N200成分 (150-300ms, 負の電位)
        n200_component = 0.0
        if 0.15 <= errp_time <= 0.3:
            n200_time = (errp_time - 0.15) / 0.15
            n200_component = -self.errp_amplitude * amplitude_factor * np.sin(np.pi * n200_time)
        
        # P300成分 (300-600ms, 正の電位)
        p300_component = 0.0
        if 0.3 <= errp_time <= 0.6:
            p300_time = (errp_time - 0.3) / 0.3
            p300_component = self.errp_amplitude * 0.8 * amplitude_factor * np.sin(np.pi * p300_time)
        
        return n200_component + p300_component
    
    def trigger_errp(self):
        """
        エラー関連電位を発生させる
        """
        self.errp_active = True
        current_time = time.time() - self.time_start
        self.errp_start_time = current_time
        
        # ErrPイベントを記録
        self.errp_events.append({
            'timestamp': current_time,
            'event_type': 'ErrP_trigger',
            'duration': 0.6  # ErrPの持続時間
        })
        
        print(f"ErrP triggered at {self.errp_start_time:.3f}s")
    
    def _streaming_thread(self):
        """
        データストリーミング用スレッド
        """
        sample_interval = 1.0 / self.sample_rate
        next_sample_time = time.time()
        
        print(f"EEG streaming started: {self.channels} channels at {self.sample_rate}Hz")
        print("Commands:")
        print("  'e' = trigger ErrP")
        print("  'q' = quit") 
        print("  's' = show current statistics")
        print("  'h' = help")
        
        while self.running:
            current_time = time.time()
            
            if current_time >= next_sample_time:
                # 経過時間を計算
                t = current_time - self.time_start
                
                # EEGサンプルを生成
                sample = self.generate_eeg_sample(t)
                
                # データを記録
                self._record_sample(sample, t)
                
                # LSLストリームに送信
                self.outlet.push_sample(sample.tolist())
                
                # 次のサンプル時刻を設定
                next_sample_time += sample_interval
            
            # 短時間スリープ（CPU使用率を下げる）
            time.sleep(0.001)
    
    def _record_sample(self, sample, timestamp):
        """
        サンプルデータを記録
        
        Args:
            sample: EEGサンプルデータ
            timestamp: タイムスタンプ
        """
        # データをリストに追加（メモリ使用量を考慮して適度に制限）
        if len(self.recording_data) < 100000:  # 最大10万サンプル（約6.7分@250Hz）
            record = {
                'timestamp': timestamp,
                'errp_active': self.errp_active
            }
            
            # 各チャンネルのデータを追加
            channel_names = [
                'Fz', 'F3', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 
                'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'Oz'
            ][:self.channels]
            
            for i, ch_name in enumerate(channel_names):
                record[ch_name] = sample[i]
            
            self.recording_data.append(record)
    
    def start_streaming(self):
        """
        ストリーミング開始
        """
        if self.running:
            print("Already streaming")
            return
            
        self.running = True
        self.time_start = time.time()
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._streaming_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_streaming(self):
        """
        ストリーミング停止
        """
        self.running = False
        if self.thread:
            self.thread.join()
        print("EEG streaming stopped")
        
        # 終了時にデータを出力
        self._export_data()
    
    def _export_data(self):
        """
        記録されたデータをファイルに出力
        """
        if not self.recording_data:
            print("No data to export")
            return
        
        # 出力ディレクトリを作成
        output_dir = "eeg_simulation_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # タイムスタンプ付きファイル名
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # EEGデータをCSVで出力
        df = pd.DataFrame(self.recording_data)
        eeg_filename = f"{output_dir}/eeg_data_{timestamp_str}.csv"
        df.to_csv(eeg_filename, index=False)
        
        # ErrPイベントをCSVで出力
        if self.errp_events:
            events_df = pd.DataFrame(self.errp_events)
            events_filename = f"{output_dir}/errp_events_{timestamp_str}.csv"
            events_df.to_csv(events_filename, index=False)
        
        # 統計情報を出力
        self._export_statistics(output_dir, timestamp_str, df)
        
        print(f"\n=== データ出力完了 ===")
        print(f"EEGデータ: {eeg_filename}")
        if self.errp_events:
            print(f"ErrPイベント: {events_filename}")
        print(f"統計情報: {output_dir}/statistics_{timestamp_str}.txt")
        print(f"総サンプル数: {len(self.recording_data)}")
        print(f"総時間: {df['timestamp'].max():.2f}秒")
        print(f"ErrPイベント数: {len(self.errp_events)}")
    
    def _export_statistics(self, output_dir, timestamp_str, df):
        """
        統計情報をテキストファイルに出力
        
        Args:
            output_dir: 出力ディレクトリ
            timestamp_str: タイムスタンプ文字列
            df: EEGデータのDataFrame
        """
        stats_filename = f"{output_dir}/statistics_{timestamp_str}.txt"
        
        channel_names = [
            'Fz', 'F3', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 
            'C4', 'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'Oz'
        ][:self.channels]
        
        with open(stats_filename, 'w', encoding='utf-8') as f:
            f.write("=== EEGシミュレーション統計情報 ===\n\n")
            f.write(f"記録開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"チャンネル数: {self.channels}\n")
            f.write(f"サンプリング周波数: {self.sample_rate} Hz\n")
            f.write(f"総サンプル数: {len(self.recording_data)}\n")
            f.write(f"総記録時間: {df['timestamp'].max():.2f} 秒\n")
            f.write(f"ErrPトリガー数: {len(self.errp_events)}\n\n")
            
            f.write("=== チャンネル別統計 ===\n")
            for ch_name in channel_names:
                if ch_name in df.columns:
                    ch_data = df[ch_name]
                    f.write(f"\n{ch_name}:\n")
                    f.write(f"  平均: {ch_data.mean():.2f} μV\n")
                    f.write(f"  標準偏差: {ch_data.std():.2f} μV\n")
                    f.write(f"  最小値: {ch_data.min():.2f} μV\n")
                    f.write(f"  最大値: {ch_data.max():.2f} μV\n")
            
            if self.errp_events:
                f.write(f"\n=== ErrPイベント詳細 ===\n")
                for i, event in enumerate(self.errp_events, 1):
                    f.write(f"イベント{i}: {event['timestamp']:.3f}秒\n")
            
            # ErrP期間中の統計
            errp_samples = df[df['errp_active'] == True]
            if not errp_samples.empty:
                f.write(f"\n=== ErrP期間中の統計 ===\n")
                f.write(f"ErrPサンプル数: {len(errp_samples)}\n")
                f.write(f"ErrP期間の平均振幅 (Fz): {errp_samples['Fz'].mean():.2f} μV\n")
                f.write(f"ErrP期間の最大振幅 (Fz): {errp_samples['Fz'].max():.2f} μV\n")
                f.write(f"ErrP期間の最小振幅 (Fz): {errp_samples['Fz'].min():.2f} μV\n")
    
    def interactive_mode(self):
        """
        インタラクティブモード
        """
        try:
            while self.running:
                user_input = input().strip().lower()
                
                if user_input == 'e':
                    self.trigger_errp()
                elif user_input == 'q':
                    break
                elif user_input == 'help' or user_input == 'h':
                    print("Commands:")
                    print("  'e' = trigger ErrP")
                    print("  'q' = quit")
                    print("  's' = show current statistics")
                    print("  'h' = help")
                elif user_input == 's':
                    self._show_current_stats()
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_streaming()
    
    def _show_current_stats(self):
        """
        現在の統計情報を表示
        """
        if not self.recording_data:
            print("No data recorded yet")
            return
        
        current_time = time.time() - self.time_start
        print(f"\n=== 現在の統計 ===")
        print(f"記録時間: {current_time:.1f}秒")
        print(f"サンプル数: {len(self.recording_data)}")
        print(f"ErrPイベント数: {len(self.errp_events)}")
        
        # 最新100サンプルの統計
        if len(self.recording_data) >= 100:
            recent_data = self.recording_data[-100:]
            fz_values = [d['Fz'] for d in recent_data]
            print(f"直近のFz平均: {np.mean(fz_values):.2f} μV")
            print(f"直近のFz標準偏差: {np.std(fz_values):.2f} μV")
        print()

def main():
    parser = argparse.ArgumentParser(description='EEG Simulator for LSL')
    parser.add_argument('--channels', type=int, default=16, help='Number of channels (default: 16)')
    parser.add_argument('--srate', type=int, default=250, help='Sample rate in Hz (default: 250)')
    parser.add_argument('--name', type=str, default='EEGSimulation_16ch', help='Stream name')
    
    args = parser.parse_args()
    
    # シミュレーター作成
    simulator = EEGSimulator(
        channels=args.channels,
        sample_rate=args.srate,
        stream_name=args.name
    )
    
    # ストリーミング開始
    simulator.start_streaming()
    
    # インタラクティブモード
    simulator.interactive_mode()

if __name__ == "__main__":
    main()