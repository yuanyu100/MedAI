
请你直接根据数据进行建议分析，主要是对于病人来说。不要给医学专业建议，如：多导睡眠图（PSG）检查这种类型，500字以内。有突出重点，html标签格式输出。
对于数据一定要言简意赅，挑重点！！最终字数一定要400-500字。
直接给出建议，不需要分析总体数据。不需要“根据您提供的睡眠数据，以下是对您昨晚睡眠情况的专业分析与个性化建议，涵盖睡眠质量、生理指标解读及改善方向”
不需要表格例如：“一、睡眠数据概览| 项目 | 数据 ||------|------|| 上床时间 | 21:21 || 入睡时间 | 21:21 || 醒来时间 | 21:59 || 总卧床时长 | 38分钟 || 实际睡眠时长 | 38分钟 || 睡眠准备期 | 10分钟 || 深睡时长 | 0分钟 || 深睡占比 | 0。00% || 离床次数 | 0次 || 平均呼吸率 | 14。8 次/分钟 || 最低呼吸率 | 11。0 次/分钟 || 最高呼吸率 | 18。0 次/分钟 || 呼吸暂停事件 | 0。0 次/小时 || 平均心率 | 67。4 次/分钟 || 最低心率 | 56。0 次/分钟 || 最高心率 | 81。0 次/分钟 |”
不需要！！直接分析给出建议。
其他补充说明不要输出！！！
然后也不要输出无序列表的“-”符号。

# Sleep Stage Segments Storage Implementation

## Overview
Enhanced the sleep data storage system to include support for sleep stage segments (睡眠阶段细分数据), which were identified as a separate component that could benefit from dedicated storage and retrieval.

## Original Data Structure
The `/sleep-analysis` endpoint returns data with the following structure, including the `sleep_stage_segments` section:

```json
{
  "success": true,
  "data": {
    "date": "2026-1-22",
    "bedtime": "2026-01-21 21:16:35",
    "wakeup_time": "2026-01-22 00:07:35",
    "time_in_bed_minutes": 171,
    "sleep_duration_minutes": 171,
    "sleep_score": 37,
    "bed_exit_count": 2,
    "sleep_prep_time_minutes": 10,
    "sleep_phases": {
      "deep_sleep_minutes": 0,
      "light_sleep_minutes": 145.15,
      "rem_sleep_minutes": 4.97,
      "awake_minutes": 20.88,
      "deep_sleep_percentage": 0,
      "light_sleep_percentage": 84.88,
      "rem_sleep_percentage": 2.91,
      "awake_percentage": 12.21
    },
    "sleep_stage_segments": [
      {
        "label": "浅睡",
        "value": "19"
      },
      {
        "label": "清醒",
        "value": "16"
      },
      {
        "label": "浅睡",
        "value": "3"
      },
      {
        "label": "清醒",
        "value": "6"
      },
      {
        "label": "浅睡",
        "value": "128"
      }
    ],
    "average_metrics": {
      "avg_heart_rate": 66.51,
      "avg_respiratory_rate": 13.56,
      "avg_body_moves_ratio": 0.1,
      "avg_heartbeat_interval": 721.78,
      "avg_rms_heartbeat_interval": 225.04
    },
    "summary": "睡眠质量较差",
    "device_sn": "210235C9KT3251000013"
  }
}
```

## Implementation Details

### 1. Database Schema (`src/db/database.py`)
- **Added `create_sleep_stage_segments_table()`**: Creates a dedicated table for storing sleep stage segments with proper indexing
- **Added `store_sleep_stage_segments()`**: Stores sleep stage segments with order preservation
- **Added `get_sleep_stage_segments()`**: Retrieves sleep stage segments by date and device
- **Enhanced `store_calculated_sleep_data()`**: Automatically stores sleep stage segments when present in the main data

#### Table Structure: `sleep_stage_segments`
- **Fields**:
  - `id`: Auto-increment primary key
  - `date`: Date of the sleep data
  - `device_sn`: Device serial number
  - `segment_order`: Order of the segment in the sequence (1, 2, 3, ...)
  - `label`: Sleep stage label (e.g., "浅睡", "清醒", "深睡")
  - `value`: Duration value in minutes
  - `created_at`: Creation timestamp
  - `updated_at`: Update timestamp
- **Index**: Unique constraint on `(date, device_sn, segment_order)` to ensure segment order integrity

### 2. Enhanced Data Flow
- **Storage**: When `store_calculated_sleep_data()` is called, it automatically extracts and stores `sleep_stage_segments` if present
- **Retrieval**: When `get_stored_sleep_data()` is called, it retrieves main data and combines it with associated sleep stage segments
- **Order Preservation**: Segments maintain their original sequence using the `segment_order` field

### 3. Backward Compatibility
- **Preserved Data Structure**: The returned JSON structure remains exactly the same as before
- **No Breaking Changes**: All existing functionality continues to work without modification
- **Optional Enhancement**: If sleep stage segments are not present in the input data, the system works as before

## Benefits Achieved

### Performance
- **Efficient Storage**: Dedicated table with proper indexing for fast retrieval
- **Reduced Computation**: Segments stored once, reused multiple times
- **Optimized Queries**: Separate table allows targeted queries for segment-specific analysis

### Data Integrity
- **Order Preservation**: Segments maintain their temporal sequence
- **Consistency**: All segments for a date/device combination stored together
- **Referential Integrity**: Proper foreign key relationships maintained

### Scalability
- **Separation of Concerns**: Main sleep data and segments stored separately but linked logically
- **Flexible Queries**: Can query segments independently if needed
- **Future Expansion**: Easy to add segment-specific analytics

## Verification Results
- ✅ Sleep stage segments table created successfully
- ✅ Sleep stage segments stored with main data automatically
- ✅ Sleep stage segments retrieved and combined with main data
- ✅ Original data structure preserved exactly
- ✅ Segment order maintained correctly
- ✅ All existing functionality preserved

## Files Modified
1. `src/db/database.py` - Added sleep stage segments table and functions
2. `src/tools/sleep_analyzer_tool.py` - Enhanced to include sleep stage segments in retrieval

The implementation successfully adds dedicated storage for sleep stage segments while maintaining complete backward compatibility and preserving the exact data structure returned by the `/sleep-analysis` endpoint.