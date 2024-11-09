try:
    import ujson as json
except ImportError:
    import json
import os
import sqlite3
import atexit
from datetime import datetime

plugin_dir = os.path.dirname(__file__)
db_path = os.path.join(plugin_dir, 'record.db')

class DatabaseHandler:
    def __init__(self):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.c = self.conn.cursor()
        self.create_table()
        self.create_indexes()

    def create_table(self):
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS draw_record (
                user_id TEXT,
                card_name TEXT,
                times TEXT,
                card_type TEXT,
                PRIMARY KEY (user_id, card_type)
            )
        ''')

        #  vote_history 表
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS vote_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_user_id TEXT,
                image_name TEXT,
                vote_result TEXT,
                vote_time TEXT,
                voters TEXT
            )
        ''')

        self.c.execute('''
            CREATE TABLE IF NOT EXISTS renamed_record (
                user_id TEXT,
                old_image_name TEXT,
                new_image_name TEXT,
                rename_time TEXT,
                PRIMARY KEY (user_id, old_image_name)
            )
        ''')

        self.c.execute('''
            CREATE TABLE IF NOT EXISTS moved_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                command TEXT,
                moved_images TEXT,
                usage_time TEXT,
                card_type TEXT
            )
        ''')
        
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS draw_history_record (
                user_id TEXT,
                card_type TEXT,
                history TEXT,
                total_count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, card_type)
            )
        ''')

        self.c.execute('''
            CREATE TABLE IF NOT EXISTS user_info (
                user_id TEXT PRIMARY KEY,
                read_help TEXT DEFAULT 0
            )
        ''')


    def create_indexes(self):
        """为 card_name 和 card_type 创建索引"""
        self.c.execute('''
            CREATE INDEX IF NOT EXISTS idx_card_name
            ON draw_record (card_name, card_type)
        ''')
        self.conn.commit()

    def get_user_info(self, user_id: str) -> dict:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            dict: 包含用户信息的字典
        """
        self.c.execute('PRAGMA table_info(user_info)')
        columns = [col[1] for col in self.c.fetchall()]
        
        self.c.execute('SELECT * FROM user_info WHERE user_id = ?', (user_id,))
        result = self.c.fetchone()
        
        return dict(zip(columns, result))
    
    def mark_help_as_read(self, user_id: str):
        """标记用户已阅读帮助"""
        # 检测用户是否已经在表中
        self.c.execute('''
            SELECT read_help FROM user_info WHERE user_id = ?
        ''', (user_id,))
        result = self.c.fetchone()

        # 如果用户未标记为已读，则进行更新
        if result is not None and result[0] != '1':
            self.c.execute('''
                UPDATE user_info
                SET read_help = ?
                WHERE user_id = ?
            ''', ('1', user_id))
            self.conn.commit()



    def update_renamed_record(self, user_id, old_image_name, new_image_name):
        """将改名后的图片记录到 renamed_images 表中，包含重命名时间"""
        rename_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.c.execute('''
            INSERT OR IGNORE INTO renamed_record (user_id, old_image_name, new_image_name, rename_time)
            VALUES (?, ?, ?, ?)
        ''', (user_id, old_image_name, new_image_name, rename_time))
        self.conn.commit()

    def get_renamed_images(self):
        """获取所有已改名的图片名和对应的新名字，返回一个包含所有记录的列表"""
        self.c.execute('SELECT old_image_name, new_image_name FROM renamed_record')
        return [(row[0], row[1]) for row in self.c.fetchall()]


    def update_draw_record(self, user_id, card_name, card_type='Wife'):
        """更新历史记录，确保一个 user_id 和 card_type 组合只能有一条记录"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.c.execute('''
            INSERT OR REPLACE INTO draw_record (user_id, card_name, times, card_type)
            VALUES (?, ?, ?, ?)
        ''', (user_id, card_name, timestamp, card_type))
        self.conn.commit()

    def get_last_trigger_time(self, user_id, card_type='Wife'):
        """获取用户最后抽老婆/老公的时间"""
        self.c.execute('SELECT times FROM draw_record WHERE user_id = ? AND card_type = ?', (user_id, card_type))
        result = self.c.fetchone()
        if result:
            return datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
        return None

    def get_card_name(self, user_id, card_type='Wife'):
        """获取用户的老婆或老公名称"""
        self.c.execute('SELECT card_name FROM draw_record WHERE user_id = ? AND card_type = ?', (user_id, card_type))
        return self.c.fetchone()

    def get_all_selected_wives_or_husbands(self, card_type='Wife'):
        """获取所有已经被抽取的老婆或老公的名字"""
        self.c.execute('SELECT card_name FROM draw_record WHERE card_type = ?', (card_type,))
        return [row[0] for row in self.c.fetchall()]

    def get_selected_wives_or_husbands_by_game(self, game_name, card_type='Wife'):
        """根据游戏名称获取已被抽取的老婆或老公，忽略大小写"""
        self.c.execute('SELECT card_name FROM draw_record WHERE LOWER(card_name) LIKE ? AND card_type = ?', (f'{game_name}_%', card_type))
        return [row[0] for row in self.c.fetchall()]

    def delete_draw_record(self, user_id, card_type='Wife'):
        """删除指定用户的老婆或老公历史记录"""
        self.c.execute('DELETE FROM draw_record WHERE user_id = ? AND card_type = ?', (user_id, card_type))
        self.conn.commit()

    def save_vote_record(self, owner_user_id, image_name, vote_result, voters):
        """保存投票记录"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.c.execute('''
            INSERT OR REPLACE INTO vote_record (owner_user_id, image_name, vote_result, vote_time, voters)
            VALUES (?, ?, ?, ?, ?)
        ''', (owner_user_id, image_name, vote_result, timestamp, voters))
        self.conn.commit()

    def get_vote_record(self, image_name):
        """检查数据库中是否有该图片的投票记录"""
        self.c.execute('SELECT id FROM vote_record WHERE image_name = ?', (image_name,))
        return self.c.fetchone()
    
    def log_moved_record(self, user_id, command, moved_images, card_type):
        """记录指令的使用情况，包括使用者、指令名、移动的图片名、时间和卡片类型"""
        usage_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        moved_images_str = ', '.join(moved_images)
        self.c.execute('''
            INSERT INTO moved_record (user_id, command, moved_images, usage_time, card_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, command, moved_images_str, usage_time, card_type))
        self.conn.commit()

    def get_moved_record(self, user_id, card_type='Wife'):
        """获取指定用户和卡片类型的指令使用记录"""
        self.c.execute('''
            SELECT command, moved_images, usage_time FROM moved_record
            WHERE user_id = ? AND card_type = ?
        ''', (user_id, card_type))
        return self.c.fetchall()
    
    def log_draw_history_record(self, user_id, card_name, card_type='Wife'):
        """记录用户每次抽卡的时间和卡片名，更新总次数"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.c.execute('SELECT history, total_count FROM draw_history_record WHERE user_id = ? AND card_type = ?', 
                    (user_id, card_type))
        result = self.c.fetchone()

        if result:
            history = json.loads(result[0]) if result[0] else []
            total_count = result[1] + 1
        else:
            history = []
            total_count = 1

        history.append({'timestamp': timestamp, 'card_name': card_name})

        self.c.execute('''
            INSERT OR REPLACE INTO draw_history_record (user_id, card_type, history, total_count)
            VALUES (?, ?, ?, ?)
        ''', (user_id, card_type, json.dumps(history, ensure_ascii=False), total_count))
        self.conn.commit()

    def get_draw_history_record(self, user_id, card_type='Wife'):
        """获取用户的抽卡历史记录和总次数"""
        self.c.execute('SELECT history, total_count FROM draw_history_record WHERE user_id = ? AND card_type = ?', 
                    (user_id, card_type))
        result = self.c.fetchone()
        if result:
            history = json.loads(result[0]) if result[0] else []
            total_count = result[1]
            return history, total_count
        return [], 0


    def close(self):
        self.conn.close()

db_handler = DatabaseHandler()
atexit.register(db_handler.close)
