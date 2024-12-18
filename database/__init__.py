try:
    import ujson as json
except ImportError:
    import json
import os
import sqlite3
import atexit
import logging
import queue
import asyncio
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager
from threading import Lock
from functools import wraps
from zhenxun.services.log import logger

class DatabaseError(Exception):
    """数据库操作异常基类
    
    属性:
        message (str): 错误信息
        error_code (int): 错误代码
        details (dict): 详细错误信息
    """
    def __init__(self, message: str, error_code: int = None, details: dict = None):
        self.message = message
        self.error_code = error_code or 5000  # 默认错误码
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        error_msg = f"数据库错误 [{self.error_code}]: {self.message}"
        if self.details:
            error_msg += f"\n详细信息: {self.details}"
        return error_msg

class ConnectionError(DatabaseError):
    """数据库连接相关错误"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, 5001, details)

class QueryError(DatabaseError):
    """数据库查询相关错误"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, 5002, details)

class TransactionError(DatabaseError):
    """数据库事务相关错误"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, 5003, details)

class ConnectionPool:
    """数据库连接池"""
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections: queue.Queue = queue.Queue(max_connections)
        self.lock = Lock()
        
        # 初始化连接池
        for _ in range(max_connections):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.connections.put(conn)

    def get_connection(self) -> sqlite3.Connection:
        try:
            return self.connections.get(timeout=5)
        except queue.Empty:
            raise ConnectionError(
                "无法获取数据库连接，连接池已满",
                {"最大连接数": self.max_connections}
            )

    def return_connection(self, connection: sqlite3.Connection):
        self.connections.put(connection)

    def close_all(self):
        while not self.connections.empty():
            conn = self.connections.get()
            conn.close()

class DatabaseHandler:
    def __init__(self, db_path: str = os.path.join(os.path.dirname(__file__), 'record.db')):
        self.db_path = db_path
        self.pool = ConnectionPool(db_path)
        self.lock = Lock()
        self.setup_database()

    @contextmanager
    def get_db_cursor(self):
        """上下文管理器处理数据库连接"""
        connection = self.pool.get_connection()
        cursor = connection.cursor()
        try:
            yield cursor
            connection.commit()
        except Exception as e:
            connection.rollback()
            logger.error(f"Database error: {str(e)}")
            raise QueryError(f"数据库操作失败", {"错误信息": str(e)})
        finally:
            cursor.close()
            self.pool.return_connection(connection)

    def retry_on_error(max_retries: int = 3, delay: float = 0.1):
        """重试装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                last_error = None
                for _ in range(max_retries):
                    try:
                        return func(self, *args, **kwargs)
                    except DatabaseError as e:
                        last_error = e
                        asyncio.sleep(delay)
                raise last_error
            return wrapper
        return decorator

    def setup_database(self):
        """初始化数据库表结构"""
        with self.get_db_cursor() as cursor:
            # 使用事务创建所有表
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS draw_record (
                    user_id TEXT,
                    card_name TEXT,
                    times TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    card_type TEXT,
                    PRIMARY KEY (user_id, card_type)
                );

                CREATE TABLE IF NOT EXISTS vote_record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_user_id TEXT,
                    image_name TEXT,
                    vote_result TEXT,
                    vote_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    voters TEXT,
                    UNIQUE(image_name)
                );

                CREATE TABLE IF NOT EXISTS renamed_record (
                    user_id TEXT,
                    old_image_name TEXT,
                    new_image_name TEXT,
                    rename_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, old_image_name)
                );

                CREATE TABLE IF NOT EXISTS moved_record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    command TEXT,
                    moved_images TEXT,
                    usage_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    card_type TEXT
                );

                CREATE TABLE IF NOT EXISTS draw_history_record (
                    user_id TEXT,
                    card_type TEXT,
                    history TEXT,
                    total_count INTEGER DEFAULT 0,
                    last_draw_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, card_type)
                );

                CREATE TABLE IF NOT EXISTS user_info (
                    user_id TEXT PRIMARY KEY,
                    read_help INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_card_name ON draw_record (card_name, card_type);
                CREATE INDEX IF NOT EXISTS idx_vote_image ON vote_record (image_name);
                CREATE INDEX IF NOT EXISTS idx_draw_history ON draw_history_record (user_id, card_type, last_draw_time);
            ''')

    @retry_on_error()
    def update_draw_record(self, user_id: str, card_name: str, card_type: str = 'Wife'):
        """更新抽卡记录并同时更新历史记录"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                INSERT OR REPLACE INTO draw_record (user_id, card_name, times, card_type)
                VALUES (?, ?, ?, ?)
            ''', (user_id, card_name, timestamp, card_type))

            cursor.execute('''
                SELECT history, total_count
                FROM draw_history_record
                WHERE user_id = ? AND card_type = ?
            ''', (user_id, card_type))
            
            result = cursor.fetchone()
            history = json.loads(result['history']) if result and result['history'] else []
            total_count = result['total_count'] + 1 if result else 1
            
            history.append({
                'timestamp': timestamp,
                'card_name': card_name
            })

            cursor.execute('''
                INSERT OR REPLACE INTO draw_history_record 
                (user_id, card_type, history, total_count)
                VALUES (?, ?, ?, ?)
            ''', (user_id, card_type, json.dumps(history), total_count))

    @retry_on_error()
    def get_user_info(self, user_id: str) -> dict:
        """获取用户信息，如果用户不存在则创建新用户"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                INSERT OR IGNORE INTO user_info (user_id, read_help)
                VALUES (?, 0)
            ''', (user_id,))
            
            cursor.execute('''
                SELECT user_id, read_help
                FROM user_info WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            return dict(row)

    @retry_on_error()
    def mark_help_as_read(self, user_id: str):
        """标记用户已阅读帮助"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                UPDATE user_info
                SET read_help = 1
                WHERE user_id = ?
                AND read_help = 0
            ''', (user_id,))

    @retry_on_error()
    def update_renamed_record(self, user_id: str, old_image_name: str, new_image_name: str):
        """将改名后的图片记录到 renamed_images 表中"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                INSERT OR IGNORE INTO renamed_record (user_id, old_image_name, new_image_name)
                VALUES (?, ?, ?)
            ''', (user_id, old_image_name, new_image_name))

    @retry_on_error()
    def get_renamed_images(self) -> List[Tuple[str, str]]:
        """获取所有已改名的图片名和对应的新名字"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT old_image_name, new_image_name FROM renamed_record')
            return [(row['old_image_name'], row['new_image_name']) for row in cursor.fetchall()]

    @retry_on_error()
    def get_last_trigger_time(self, user_id: str, card_type: str = 'Wife') -> Optional[datetime]:
        """获取用户最后抽卡时间"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT times FROM draw_record WHERE user_id = ? AND card_type = ?', 
                         (user_id, card_type))
            result = cursor.fetchone()
            if result:
                return datetime.strptime(result['times'], '%Y-%m-%d %H:%M:%S')
            return None
        
    
    @retry_on_error()
    def get_draw_history_record(self, user_id: str, card_type: str = 'Wife') -> Tuple[List[Dict[str, str]], int]:
        """获取用户的抽卡历史记录和总次数"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                SELECT history, total_count
                FROM draw_history_record
                WHERE user_id = ? AND card_type = ?
            ''', (user_id, card_type))
            
            result = cursor.fetchone()
            if result:
                history = json.loads(result['history']) if result['history'] else []
                return history, result['total_count']
            return [], 0
        
    @retry_on_error()
    def save_vote_record(self, owner_user_id: str, image_name: str, 
                        vote_result: str, voters: Dict[str, List[str]]):
        """保存投票记录"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                INSERT OR REPLACE INTO vote_record 
                (owner_user_id, image_name, vote_result, voters)
                VALUES (?, ?, ?, ?)
            ''', (owner_user_id, image_name, vote_result, 
                  json.dumps(voters, ensure_ascii=False)))
            
    @retry_on_error()
    def update_renamed_record(self, user_id: str, old_image_name: str, new_image_name: str) -> None:
        """将改名后的图片记录到 renamed_images 表中，包含重命名时间"""
        with self.get_db_cursor() as cursor:
            rename_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT OR IGNORE INTO renamed_record (user_id, old_image_name, new_image_name, rename_time)
                VALUES (?, ?, ?, ?)
            ''', (user_id, old_image_name, new_image_name, rename_time))

    @retry_on_error()
    def get_renamed_images(self) -> List[Tuple[str, str]]:
        """获取所有已改名的图片名和对应的新名字，返回一个包含所有记录的列表"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT old_image_name, new_image_name FROM renamed_record')
            return [(row['old_image_name'], row['new_image_name']) for row in cursor.fetchall()]

    @retry_on_error()
    def get_last_trigger_time(self, user_id: str, card_type: str = 'Wife') -> Optional[datetime]:
        """获取用户最后抽老婆/老公的时间"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT times FROM draw_record WHERE user_id = ? AND card_type = ?', 
                        (user_id, card_type))
            result = cursor.fetchone()
            if result and result['times']:
                return datetime.strptime(result['times'], '%Y-%m-%d %H:%M:%S')
            return None

    @retry_on_error()
    def get_card_name(self, user_id: str, card_type: str = 'Wife') -> Optional[str]:
        """获取用户的老婆或老公名称"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT card_name FROM draw_record WHERE user_id = ? AND card_type = ?', 
                        (user_id, card_type))
            result = cursor.fetchone()
            return result['card_name'] if result else None

    @retry_on_error()
    def get_all_selected_wives_or_husbands(self, card_type: str = 'Wife') -> List[str]:
        """获取所有已经被抽取的老婆或老公的名字"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT card_name FROM draw_record WHERE card_type = ?', (card_type,))
            return [row['card_name'] for row in cursor.fetchall()]

    @retry_on_error()
    def get_selected_wives_or_husbands_by_game(self, game_name: str, card_type: str = 'Wife') -> List[str]:
        """根据游戏名称获取已被抽取的老婆或老公，忽略大小写"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                SELECT card_name FROM draw_record 
                WHERE LOWER(card_name) LIKE ? AND card_type = ?
            ''', (f'{game_name.lower()}_%', card_type))
            return [row['card_name'] for row in cursor.fetchall()]

    @retry_on_error()
    def delete_draw_record(self, user_id: str, card_type: str = 'Wife') -> None:
        """删除指定用户的老婆或老公历史记录"""
        with self.get_db_cursor() as cursor:
            cursor.execute('DELETE FROM draw_record WHERE user_id = ? AND card_type = ?', 
                        (user_id, card_type))

    @retry_on_error()
    def save_vote_record(self, owner_user_id: str, image_name: str, 
                        vote_result: str, voters: str) -> None:
        """保存投票记录"""
        with self.get_db_cursor() as cursor:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                INSERT OR REPLACE INTO vote_record 
                (owner_user_id, image_name, vote_result, vote_time, voters)
                VALUES (?, ?, ?, ?, ?)
            ''', (owner_user_id, image_name, vote_result, timestamp, voters))

    @retry_on_error()
    def get_vote_record(self, image_name: str) -> Optional[int]:
        """检查数据库中是否有该图片的投票记录"""
        with self.get_db_cursor() as cursor:
            cursor.execute('SELECT id FROM vote_record WHERE image_name = ?', (image_name,))
            result = cursor.fetchone()
            return result['id'] if result else None

    @retry_on_error()
    def log_moved_record(self, user_id: str, command: str, 
                        moved_images: List[str], card_type: str) -> None:
        """记录指令的使用情况，包括使用者、指令名、移动的图片名、时间和卡片类型"""
        with self.get_db_cursor() as cursor:
            moved_images_str = ', '.join(moved_images)
            cursor.execute('''
                INSERT INTO moved_record (user_id, command, moved_images, card_type)
                VALUES (?, ?, ?, ?)
            ''', (user_id, command, moved_images_str, card_type))

    @retry_on_error()
    def get_moved_record(self, user_id: str, card_type: str = 'Wife') -> List[Tuple[str, str, str]]:
        """获取指定用户和卡片类型的指令使用记录"""
        with self.get_db_cursor() as cursor:
            cursor.execute('''
                SELECT command, moved_images, usage_time 
                FROM moved_record
                WHERE user_id = ? AND card_type = ?
            ''', (user_id, card_type))
            return [(row['command'], row['moved_images'], row['usage_time']) 
                    for row in cursor.fetchall()]
        
    @retry_on_error()
    def log_draw_history_record(self, user_id: str, card_name: str, card_type: str = 'Wife') -> None:
        """
        记录用户每次抽卡的时间和卡片名，更新总次数
        
        Args:
            user_id (str): 用户ID
            card_name (str): 卡片名称
            card_type (str, optional): 卡片类型，默认为'Wife'
            
        Returns:
            None
        """
        with self.get_db_cursor() as cursor:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                SELECT history, total_count 
                FROM draw_history_record 
                WHERE user_id = ? AND card_type = ?
            ''', (user_id, card_type))
            
            result = cursor.fetchone()
            
            if result:
                history = json.loads(result['history']) if result['history'] else []
                total_count = result['total_count'] + 1
            else:
                history = []
                total_count = 1
            
            history.append({
                'timestamp': timestamp,
                'card_name': card_name
            })
            
            cursor.execute('''
                INSERT OR REPLACE INTO draw_history_record 
                (user_id, card_type, history, total_count)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                card_type,
                json.dumps(history, ensure_ascii=False),
                total_count
            ))

    def close(self):
        """关闭数据库连接池"""
        self.pool.close_all()

db_handler = DatabaseHandler()
atexit.register(db_handler.close)