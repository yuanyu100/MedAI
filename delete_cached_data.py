from src.db.database import get_db_manager

# 删除数据库中的缓存数据
db_manager = get_db_manager()
date = "2026-02-02"
device_sn = "210235C9KT3251000013"

print("=== 删除数据库缓存数据 ===")

# 直接执行SQL删除语句
try:
    # 删除指定日期和设备的缓存数据
    delete_query = "DELETE FROM calculated_sleep_data WHERE date = :date AND device_sn = :device_sn"
    params = {'date': date, 'device_sn': device_sn}
    db_manager.execute_command(delete_query, params)
    print(f"✅ 成功删除缓存数据: 日期={date}, 设备={device_sn}")
    
    # 验证删除结果
    cached_data = db_manager.get_calculated_sleep_data(date, device_sn)
    print(f"删除后缓存数据条数: {len(cached_data)}")
    
    if len(cached_data) == 0:
        print("✅ 验证成功：缓存数据已完全删除")
    else:
        print("❌ 验证失败：缓存数据未完全删除")
        
except Exception as e:
    print(f"❌ 删除缓存数据失败: {str(e)}")

print("\n=== 操作完成 ===")
print("下次请求将强制重新计算睡眠数据")
