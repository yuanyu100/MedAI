from src.db.database import get_db_manager

# 检查数据库中的缓存数据
db_manager = get_db_manager()
date = "2026-02-02"
device_sn = "210235C9KT3251000013"

print("=== 检查数据库缓存数据 ===")

# 获取已计算的睡眠数据
cached_data = db_manager.get_calculated_sleep_data(date, device_sn)
print(f"缓存数据条数: {len(cached_data)}")

if not cached_data.empty:
    record = cached_data.to_dict('records')[0]
    print(f"\n缓存记录详情:")
    print(f"  date: {record.get('date')}")
    print(f"  device_sn: {record.get('device_sn')}")
    print(f"  bedtime: {record.get('bedtime')}")
    print(f"  wakeup_time: {record.get('wakeup_time')}")
    print(f"  sleep_prep_time_minutes: {record.get('sleep_prep_time_minutes')}")
    print(f"  sleep_duration_minutes: {record.get('sleep_duration_minutes')}")
    print(f"  sleep_score: {record.get('sleep_score')}")

# 尝试删除缓存数据，强制重新计算
print("\n=== 删除缓存数据 ===")
db_manager.delete_calculated_sleep_data(date, device_sn)
print("缓存数据已删除，下次请求将重新计算")
