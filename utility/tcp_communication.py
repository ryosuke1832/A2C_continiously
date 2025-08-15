# tcp_communication.py
# A2C_UnityROS_CSV_classificationの通信部分のみを分離

import socket
import json
import time
import threading
from typing import Dict, Any, Optional, Callable

class UnityTcpServer:
    """
    Unity通信用のシンプルなTCPサーバー
    現在使用しているコードから通信部分のみを抽出
    """
    
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
        self.client_socket = None
        
        # メッセージハンドラー - 外部から設定可能
        self.message_handler: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None
        
        print(f"🚀 UnityTcpServer初期化完了")
        print(f"📡 接続先: {self.host}:{self.port}")

    def set_message_handler(self, handler: Callable[[str, Dict[str, Any]], Dict[str, Any]]):
        """メッセージハンドラーを設定"""
        self.message_handler = handler

    def connect_to_unity(self):
        """
        Unityシミュレータに接続
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"🚀 TCPサーバー開始: {self.host}:{self.port}")
            print("⏳ Unity接続待機中...")
            
            client_socket, address = self.socket.accept()
            print(f"✅ Unity接続: {address}")
            self.is_connected = True
            self.client_socket = client_socket
            return client_socket
            
        except Exception as e:
            print(f"❌ 接続エラー: {e}")
            return None

    def process_unity_data(self, client_socket):
        """
        Unityからのデータを処理（元のコードから抽出）
        """
        buffer = ""  # バッファを追加してJSON解析を安定化
        
        while self.is_connected:
            try:
                raw_data = client_socket.recv(1024).decode('utf-8')
                if not raw_data:
                    break
                
                buffer += raw_data
                
                # 完全なJSONメッセージを探す
                while True:
                    try:
                        # 改行で分割して個別のJSONメッセージを処理
                        if '\n' in buffer:
                            json_line, buffer = buffer.split('\n', 1)
                        else:
                            json_line = buffer
                            buffer = ""
                        
                        if not json_line.strip():
                            break
                            
                        # JSONパース
                        parsed_data = json.loads(json_line.strip())
                        print(f"📥 受信データ: {parsed_data}")
                        
                        # メッセージタイプに応じて処理
                        message_type = parsed_data.get('type', 'unknown')
                        
                        # デフォルト応答
                        if message_type == 'ping':
                            response = {
                                "type": "pong",
                                "message": "サーバー動作中",
                                "timestamp": time.time()
                            }
                        else:
                            # カスタムハンドラーがあれば使用
                            if self.message_handler:
                                response = self.message_handler(message_type, parsed_data)
                            else:
                                # デフォルト応答
                                response = {
                                    "type": "ack",
                                    "message": f"メッセージ受信: {message_type}",
                                    "timestamp": time.time()
                                }
                        
                        # 応答送信
                        if response:
                            response_json = json.dumps(response) + '\n'
                            client_socket.send(response_json.encode('utf-8'))
                        
                        if '\n' not in buffer:
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON解析エラー（スキップ）: {e}")
                        if '\n' not in buffer:
                            buffer = ""
                            break
                        continue
                        
            except Exception as e:
                print(f"❌ データ処理エラー: {e}")
                break
        
        print("🔌 Unity接続終了")
        client_socket.close()

    def send_message(self, message: Dict[str, Any]):
        """メッセージをUnityに送信"""
        if self.client_socket and self.is_connected:
            try:
                message_json = json.dumps(message) + '\n'
                self.client_socket.send(message_json.encode('utf-8'))
                return True
            except Exception as e:
                print(f"❌ メッセージ送信エラー: {e}")
                return False
        return False

    def run(self):
        """
        メインループ実行
        """
        client_socket = self.connect_to_unity()
        if client_socket:
            self.process_unity_data(client_socket)

    def stop(self):
        """サーバー停止"""
        self.is_connected = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        print("🛑 TCPサーバー停止")

# 使用例: 元のA2C分類器の通信部分をそのまま使用
class A2CMessageHandler:
    """
    元のA2Cプロジェクト用のメッセージハンドラー
    """
    
    def __init__(self):
        self.current_episode = 0
        self.current_step = 0
        self.start_time = time.time()
        print("🤖 A2CMessageHandler初期化完了")

    def handle_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        元のコードのメッセージ処理ロジック
        """
        if message_type == 'can_state':
            # 缶の状態データを処理
            current_force = data['current_force']
            accumulated_force = data['accumulated_force']
            is_crushed = data['is_crushed']
            timestamp = data['timestamp']
            
            print(f"🎯 缶状態: 力={current_force:.2f}N, 潰れ={is_crushed}")
            
            # 推奨アクション生成（簡単な例）
            if is_crushed:
                recommended_force = 0.0
                action = "緊急停止"
            elif current_force > 20.0:
                recommended_force = max(current_force - 2.0, 5.0)
                action = "力を減少"
            elif current_force < 5.0:
                recommended_force = min(current_force + 1.0, 15.0)
                action = "力を増加"
            else:
                recommended_force = current_force
                action = "現状維持"
            
            self.current_step += 1
            
            return {
                "type": "action_response",
                "recommended_force": recommended_force,
                "action": action,
                "current_step": self.current_step,
                "episode": self.current_episode,
                "timestamp": time.time()
            }
        
        elif message_type == 'reset':
            print("🔄 シミュレーション リセット")
            self.current_step = 0
            return {
                "type": "reset_complete",
                "message": "リセット完了",
                "timestamp": time.time()
            }
        
        elif message_type == 'episode_end':
            print(f"🏁 エピソード {self.current_episode} 終了")
            response = {
                "type": "episode_complete",
                "episode": self.current_episode,
                "total_steps": self.current_step,
                "timestamp": time.time()
            }
            self.current_episode += 1
            self.current_step = 0
            return response
        
        # デフォルト応答
        return {
            "type": "ack",
            "message": f"メッセージ受信: {message_type}",
            "timestamp": time.time()
        }

# 使用例: カスタムメッセージハンドラー
class CustomMessageHandler:
    """
    他のプロジェクト用のカスタムハンドラー例
    """
    
    def __init__(self):
        self.message_count = 0
        print("🔧 CustomMessageHandler初期化完了")

    def handle_message(self, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        カスタムメッセージ処理
        """
        self.message_count += 1
        
        if message_type == 'sensor_data':
            # センサーデータ処理の例
            position = data.get('position', {})
            velocity = data.get('velocity', {})
            
            print(f"📊 センサーデータ: pos={position}, vel={velocity}")
            
            return {
                "type": "sensor_ack",
                "processed": True,
                "message_count": self.message_count,
                "timestamp": time.time()
            }
        
        elif message_type == 'chat_message':
            # チャットメッセージ処理の例
            username = data.get('username', 'Anonymous')
            content = data.get('content', '')
            
            print(f"💬 チャット: {username}: {content}")
            
            return {
                "type": "chat_broadcast",
                "username": username,
                "content": content,
                "timestamp": time.time()
            }
        
        # デフォルト応答
        return {
            "type": "ack",
            "received_type": message_type,
            "message_count": self.message_count,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    # 使用例1: 元のA2Cプロジェクトと同じ使い方
    print("🎯 A2C通信サーバー開始")
    server = UnityTcpServer('127.0.0.1', 12345)
    a2c_handler = A2CMessageHandler()
    server.set_message_handler(a2c_handler.handle_message)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n⏹️ サーバー停止中...")
        server.stop()

# 他のプロジェクトでの使用例:
"""
# センサーデータ処理サーバー
server = UnityTcpServer('localhost', 8080)
custom_handler = CustomMessageHandler()
server.set_message_handler(custom_handler.handle_message)
server.run()

# チャットサーバー
server = UnityTcpServer('localhost', 9999)
chat_handler = CustomMessageHandler()
server.set_message_handler(chat_handler.handle_message)
server.run()
"""