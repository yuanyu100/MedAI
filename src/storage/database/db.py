import os
import pymysql
from typing import Optional


def get_db_url() -> Optional[str]:
    """
    获取数据库连接URL
    优先使用环境变量，否则返回MySQL默认连接信息
    """
    # 优先使用环境变量
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # 使用默认MySQL配置
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "040726fan")
    database = os.getenv("MYSQL_DATABASE", "ai")  # 使用ai数据库
    
    # 构建MySQL连接字符串
    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    
    return db_url


def test_mysql_connection():
    """
    测试MySQL连接
    """
    try:
        host = os.getenv("MYSQL_HOST", "localhost")
        port = int(os.getenv("MYSQL_PORT", "3306"))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "040726fan")
        database = os.getenv("MYSQL_DATABASE", "bed_monitoring")
        
        # 首先连接到MySQL服务器（不指定数据库）
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset='utf8mb4'
        )
        
        # 检查并创建数据库
        cursor = connection.cursor()
        cursor.execute("SHOW DATABASES;")
        databases = [db[0] for db in cursor.fetchall()]
        
        if database not in databases:
            print(f"数据库 {database} 不存在，正在创建...")
            cursor.execute(f"CREATE DATABASE {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
            print(f"数据库 {database} 创建成功")
        
        cursor.close()
        connection.close()
        
        # 现在连接到指定数据库
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
        
        print("✓ MySQL数据库连接成功")
        connection.close()
        return True
        
    except Exception as e:
        print(f"✗ MySQL数据库连接失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 测试连接
    test_mysql_connection()