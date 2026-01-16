# 数据库配置说明

## MySQL数据库配置

项目现在支持使用MySQL数据库作为持久化存储。以下是配置说明：

### 默认配置

- **主机**: localhost
- **端口**: 3306
- **用户名**: root
- **密码**: 040726fan
- **数据库名**: bed_monitoring

### 环境变量配置

可通过环境变量自定义数据库配置：

```bash
export MYSQL_HOST=localhost      # MySQL服务器地址
export MYSQL_PORT=3306          # MySQL端口
export MYSQL_USER=root          # 用户名
export MYSQL_PASSWORD=040726fan # 密码
export MYSQL_DATABASE=bed_monitoring  # 数据库名
```

### 自动数据库创建

系统会自动检测并创建数据库（如果不存在）：
1. 首先连接到MySQL服务器（不指定数据库）
2. 检查目标数据库是否存在
3. 如果不存在，则自动创建数据库
4. 连接到指定数据库进行后续操作

### 依赖包

项目使用以下依赖包支持MySQL：

- `PyMySQL`: Python MySQL客户端
- `mysql-connector-python`: MySQL官方连接器
- `langgraph-checkpoint-mysql`: LangGraph的MySQL检查点支持

### 数据库连接流程

1. 系统优先尝试使用MySQL数据库
2. 如果MySQL连接失败，则自动回退到内存存储（MemorySaver）
3. 系统使用PyMySQLSaver（基于上下文管理器）进行数据库操作
4. 所有会话数据和状态将被持久化到MySQL数据库中

### 连接验证

系统在启动时会验证MySQL连接：
- 如果MySQL配置正确，系统会记录连接验证成功
- 如果连接失败，系统会回退到内存存储模式
- 即使在内存模式下，MySQL连接也会在需要时按需创建

### 故障排除

- **连接失败**: 确认MySQL服务正在运行且网络可达
- **权限问题**: 确认用户具有创建数据库和表的权限
- **字符集问题**: 数据库使用utf8mb4字符集确保中文支持
- **依赖问题**: 确保已安装langgraph-checkpoint-mysql包