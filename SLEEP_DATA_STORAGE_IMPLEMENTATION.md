# Sleep and Physiological Data Storage Implementation

## Overview

This implementation adds database storage functionality for calculated sleep analysis results and physiological metrics obtained from the `/sleep-analysis` and `/physiological-analysis` endpoints. The system now stores computed results in a structured database table and retrieves them when available, reducing redundant calculations.

## Key Features Implemented

### 1. Database Table Structure
- **Table Name**: `calculated_sleep_data`
- **Purpose**: Stores calculated sleep and physiological analysis results
- **Key Fields**:
  - `date`: Date of the analysis (DATE)
  - `device_sn`: Device serial number (VARCHAR)
  - `bedtime`: Bedtime (DATETIME)
  - `wakeup_time`: Wake-up time (DATETIME)
  - `time_in_bed_minutes`: Time spent in bed (DECIMAL)
  - `sleep_duration_minutes`: Actual sleep duration (DECIMAL)
  - `sleep_score`: Sleep quality score (INT)
  - `bed_exit_count`: Number of times getting out of bed (INT)
  - `sleep_prep_time_minutes`: Time to fall asleep (DECIMAL)
  - Sleep phase metrics (deep, light, REM, awake durations and percentages)
  - Heart rate metrics (average, min, max, HRV score)
  - Respiratory metrics (average, min, max, apnea events, health score)
  - Timestamps for tracking creation/updates

### 2. Database Manager Functions
- `create_calculated_sleep_data_table()`: Creates the storage table with proper indexes
- `store_calculated_sleep_data()`: Stores calculated results with duplicate key handling (ON DUPLICATE KEY UPDATE)
- `get_calculated_sleep_data()`: Retrieves stored results by date and device

### 3. Tool Functions Added
#### In `sleep_analyzer_tool.py`:
- `store_calculated_sleep_data()`: Tool to store sleep analysis results
- `get_stored_sleep_data()`: Tool to retrieve stored sleep data

#### In `physiological_analyzer_tool.py`:
- `store_calculated_physiological_data()`: Tool to store physiological analysis results
- `get_stored_physiological_data()`: Tool to retrieve stored physiological data

### 4. Service Layer Updates
#### In `sleep_analysis_service.py`:
- `run_sleep_analysis_with_formatted_time()`: Updated to check database first, compute if needed, and store results
- `get_formatted_sleep_time_summary()`: Updated to check database first, compute if needed, and store results

### 5. API Endpoint Updates
#### In `fixed_api_server.py`:
- `/sleep-analysis` endpoint: Updated to check database first, compute if needed, and store results
- `/physiological-analysis` endpoint: Updated to check database first, compute if needed, and store results

## How It Works

1. **Request Flow**:
   - When a request comes to `/sleep-analysis` or `/physiological-analysis`
   - System first checks if results already exist in the database
   - If found, returns stored results immediately
   - If not found, computes the analysis from raw data
   - Stores the computed results in the database
   - Returns the results to the user

2. **Duplicate Handling**:
   - Uses `ON DUPLICATE KEY UPDATE` clause
   - Prevents duplicate entries for the same date/device combination
   - Updates existing records with new values when duplicates occur

3. **Data Mapping**:
   - Maps sleep analysis results to appropriate database fields
   - Maps physiological analysis results to appropriate database fields
   - Preserves all key metrics including bedtime, sleep duration, heart rate, respiratory rate, etc.

## Benefits

1. **Performance**: Reduces redundant calculations by reusing stored results
2. **Consistency**: Ensures consistent results for the same input data
3. **Scalability**: Prevents repeated heavy computations
4. **Data Persistence**: Maintains historical analysis results
5. **Efficiency**: Optimizes API response times

## Metrics Stored

The system stores all the metrics mentioned in the original request:
- 上床时间 (Bedtime): 21:16
- 入睡时间 (Sleep onset time): 21:16
- 醒来时间 (Wake-up time): 00:07
- 总卧床时长 (Total time in bed): 2小时51分钟 (2.85 hours)
- 实际睡眠时长 (Actual sleep duration): 2小时51分钟 (2.85 hours)
- 睡眠准备期 (Sleep prep time): 10 minutes
- 深睡时长 (Deep sleep duration): 0.0 minutes (0.0%)
- 中间离床次数 (Number of bed exits): 2 times
- 平均呼吸率 (Average respiratory rate): 14.2次/分钟
- 最低呼吸率 (Minimum respiratory rate): 9.0次/分钟
- 最高呼吸率 (Maximum respiratory rate): 19.0次/分钟
- 呼吸暂停事件 (Apnea events): 0.0次/小时
- 平均心率 (Average heart rate): 69.8次/分钟
- 最低心率 (Minimum heart rate): 54.0次/分钟
- 最高心率 (Maximum heart rate): 104.0次/分钟

## Testing

The implementation has been thoroughly tested with:
- Direct database storage and retrieval
- Duplicate key handling
- End-to-end functionality verification
- Proper error handling

All tests pass successfully, confirming that the storage system works as intended.