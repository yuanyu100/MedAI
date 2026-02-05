#!/usr/bin/env python3
"""
Initialize the sleep and physiological data storage system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.db.database import get_db_manager

def initialize_sleep_data_storage():
    """Initialize the calculated sleep data storage system"""
    print("Initializing sleep and physiological data storage...")
    
    try:
        # Get database manager instance
        db_manager = get_db_manager()
        
        # Create the calculated sleep data table (this also handles physiological data)
        print("Creating calculated sleep data table...")
        db_manager.create_calculated_sleep_data_table()
        print("✓ Calculated sleep data table created successfully!")
        
        # Test inserting sample data
        print("\nTesting with sample data...")
        sample_data = {
            'date': '2024-12-20',
            'device_sn': 'TEST_DEVICE_001',
            'bedtime': '2024-12-20 21:16:00',
            'wakeup_time': '2024-12-21 00:07:00',
            'time_in_bed_minutes': 171.0,  # 2小时51分钟
            'sleep_duration_minutes': 171.0,  # 2小时51分钟
            'sleep_score': 75,
            'bed_exit_count': 2,
            'sleep_prep_time_minutes': 10,
            'sleep_phases': {
                'deep_sleep_minutes': 0.0,
                'light_sleep_minutes': 120.0,
                'rem_sleep_minutes': 30.0,
                'awake_minutes': 21.0,
                'deep_sleep_percentage': 0.0,
                'light_sleep_percentage': 70.0,
                'rem_sleep_percentage': 18.0,
                'awake_percentage': 12.0
            },
            'average_metrics': {
                'avg_heart_rate': 69.8,
                'avg_respiratory_rate': 14.2,
                'min_heart_rate': 54.0,
                'max_heart_rate': 104.0,
                'min_respiratory_rate': 9.0,
                'max_respiratory_rate': 19.0
            },
            'respiratory_metrics': {
                'apnea_count': 0.0
            }
        }
        
        # Store the sample data
        print("Storing sample data to database...")
        db_manager.store_calculated_sleep_data(sample_data)
        print("✓ Sample data stored successfully!")
        
        # Retrieve the stored data
        print("Retrieving stored data from database...")
        retrieved_data = db_manager.get_calculated_sleep_data('2024-12-20', 'TEST_DEVICE_001')
        print(f"✓ Retrieved {len(retrieved_data)} records from database")
        
        if not retrieved_data.empty:
            print("\nSample data successfully stored and retrieved:")
            print(f"Date: {retrieved_data.iloc[0]['date']}")
            print(f"Device: {retrieved_data.iloc[0]['device_sn']}")
            print(f"Bedtime: {retrieved_data.iloc[0]['bedtime']}")
            print(f"Wake-up time: {retrieved_data.iloc[0]['wakeup_time']}")
            print(f"Time in bed: {retrieved_data.iloc[0]['time_in_bed_minutes']} minutes")
            print(f"Sleep duration: {retrieved_data.iloc[0]['sleep_duration_minutes']} minutes")
            print(f"Deep sleep: {retrieved_data.iloc[0]['deep_sleep_minutes']} minutes ({retrieved_data.iloc[0]['deep_sleep_percentage']}%)")
            print(f"Avg heart rate: {retrieved_data.iloc[0]['avg_heart_rate']} bpm")
            print(f"Avg respiratory rate: {retrieved_data.iloc[0]['avg_respiratory_rate']} breaths/min")
            print(f"Apnea count: {retrieved_data.iloc[0]['apnea_count']}")
        
        print("\n✓ Sleep and physiological data storage system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error initializing sleep data storage: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = initialize_sleep_data_storage()
    if success:
        print("\n🎉 Initialization completed successfully!")
    else:
        print("\n💥 Initialization failed!")
        sys.exit(1)