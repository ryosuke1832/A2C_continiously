# unity_communication_test_robust.py
import socket
import json
import threading
import time
import signal
import sys
from datetime import datetime

class UnityCommunicationTest:
    def __init__(self, host='127.0.0.1', port=12345):  # localhostの代わりに明示的IP
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None
        self.running = False
        
        # 受信データ統計
        self.message_count = 0
        self.start_time = None
        
        # 状態履歴（最新10件）
        self.state_history = []
        
        # Ctrl+C対応
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Ctrl+C処理"""
        print("\n⏹️ Ctrl+C検出 - サーバーを停止しています...")
        self.running = False
        if self.socket:
            self.socket.close()
        sys.exit(0)
    
    def start_server(self):
        """サーバー開始"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(1.0)  # 1秒タイムアウト（Ctrl+C対応）
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            print("=" * 60)
            print("🚀 Unity通信テストサーバー開始")
            print(f"📡 接続先: {self.host}:{self.port}")
            print("⏳ Unityクライアントからの接続を待機中...")
            print("💡 Ctrl+C で停止できます")
            print("=" * 60)
            
            self.running = True
            self.start_time = time.time()
            
            # 接続待機ループ（タイムアウト付き）
            while self.running:
                try:
                    self.client_socket, addr = self.socket.accept()
                    print(f"✅ Unityクライアント接続成功: {addr}")
                    print(f"⏰ 接続時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("-" * 60)
                    
                    # 通信ループ開始
                    self.communication_loop()
                    break
                    
                except socket.timeout:
                    # タイムアウト時はCtrl+Cチェックのために継続
                    continue
                except Exception as e:
                    if self.running:  # 正常終了中でない場合のみエラー表示
                        print(f"❌ 接続受付エラー: {e}")
                    break
            
        except Exception as e:
            print(f"❌ サーバー開始エラー: {e}")
        finally:
            self.cleanup()
    
    def communication_loop(self):
        """メイン通信ループ"""
        print("📡 通信ループ開始 - データ受信待機中...")
        print("💡 Unity側でPキーを押してPingテストを実行してください")
        print()
        
        # クライアントソケットにもタイムアウト設定
        if self.client_socket:
            self.client_socket.settimeout(1.0)
        
        while self.running:
            try:
                # Unityからデータ受信（タイムアウト付き）
                data = self.client_socket.recv(4096)
                if not data:
                    print("⚠️ クライアント切断検出")
                    break
                
                # JSONデコード
                try:
                    message = json.loads(data.decode('utf-8'))
                    
                    # メッセージ表示
                    self.display_message(message)
                    
                    # 簡単な応答を送信
                    response = self.create_simple_response(message)
                    if response:
                        response_json = json.dumps(response)
                        self.client_socket.send(response_json.encode('utf-8'))
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析エラー: {e}")
                    print(f"   受信データ: {data.decode('utf-8', errors='ignore')}")
                    
            except socket.timeout:
                # タイムアウトは正常（Ctrl+Cチェック用）
                continue
            except Exception as e:
                if self.running:
                    print(f"❌ 通信エラー: {e}")
                break
    
    def display_message(self, message):
        """受信メッセージの詳細表示"""
        self.message_count += 1
        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        print(f"📨 メッセージ #{self.message_count} ({current_time})")
        print("-" * 40)
        
        msg_type = message.get('type', 'unknown')
        print(f"🏷️  タイプ: {msg_type}")
        
        if msg_type == 'can_state':
            self.display_can_state(message)
        elif msg_type == 'ping':
            print("🏓 Pingメッセージ受信")
            print("💬 Unity側との通信が正常です！")
        elif msg_type == 'test':
            print("🧪 テストメッセージ受信")
        else:
            print(f"📋 メッセージ内容: {json.dumps(message, indent=2, ensure_ascii=False)}")
        
        print("=" * 40)
        print()
    
    def display_can_state(self, message):
        """アルミ缶状態の詳細表示"""
        is_crushed = message.get('is_crushed', False)
        current_force = message.get('current_force', 0.0)
        accumulated_force = message.get('accumulated_force', 0.0)
        timestamp = message.get('timestamp', 0.0)
        
        # 状態アイコン
        status_icon = "💔" if is_crushed else "✅"
        force_icon = "⚠️" if current_force > 15.0 else "🔧" if current_force > 0 else "⏸️"
        
        print(f"{status_icon} 缶の状態: {'つぶれた' if is_crushed else '正常'}")
        print(f"{force_icon} 現在の力: {current_force:.2f} N")
        print(f"📊 蓄積力: {accumulated_force:.2f} N")
        print(f"⏰ Unity時刻: {timestamp:.3f}")
        
        # 力の危険レベル表示
        if current_force > 0:
            danger_level = min(current_force / 20.0, 1.0)
            bar_length = 20
            filled_length = int(bar_length * danger_level)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            if danger_level < 0.3:
                level_text = "安全"
                level_color = "🟢"
            elif danger_level < 0.7:
                level_text = "注意"
                level_color = "🟡"
            else:
                level_text = "危険"
                level_color = "🔴"
            
            print(f"📈 力レベル: {level_color} [{bar}] {danger_level*100:.1f}% ({level_text})")
        
        # 状態履歴に追加
        self.state_history.append({
            'is_crushed': is_crushed,
            'current_force': current_force,
            'accumulated_force': accumulated_force,
            'timestamp': timestamp,
            'receive_time': time.time()
        })
        
        if len(self.state_history) > 10:
            self.state_history.pop(0)
    
    def create_simple_response(self, message):
        """簡単な応答作成"""
        msg_type = message.get('type')
        
        response = {
            'type': 'ack',
            'timestamp': time.time(),
            'original_type': msg_type
        }
        
        if msg_type == 'ping':
            response['type'] = 'pong'
            response['message'] = 'Python側で正常に受信'
        elif msg_type == 'can_state':
            response['message'] = 'アルミ缶状態受信完了'
        elif msg_type == 'test':
            response['message'] = 'テストメッセージ受信完了'
        
        return response
    
    def print_statistics(self):
        """統計情報の表示"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            messages_per_second = self.message_count / elapsed_time if elapsed_time > 0 else 0
            
            print("\n" + "=" * 50)
            print("📊 通信統計")
            print("-" * 50)
            print(f"⏱️  通信時間: {elapsed_time:.2f} 秒")
            print(f"📮 総メッセージ数: {self.message_count}")
            print(f"📈 メッセージ/秒: {messages_per_second:.2f}")
            
            if self.message_count > 0:
                print("✅ 通信は正常に動作しました！")
            else:
                print("⚠️ メッセージが受信されませんでした")
            print("=" * 50)
    
    def cleanup(self):
        """リソースクリーンアップ"""
        self.running = False
        
        print("\n🔄 クリーンアップ中...")
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
                
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        # 統計表示
        self.print_statistics()
        
        print("🛑 サーバー停止完了")

# メイン実行
if __name__ == "__main__":
    print("Unity通信テストサーバー (改良版)")
    print("Ctrl+C で安全に終了できます")
    print()
    
    server = UnityCommunicationTest()
    server.start_server()