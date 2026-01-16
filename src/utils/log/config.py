"""
Application configuration
"""
import os
from pathlib import Path

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

import platform

# Determine appropriate log directory based on OS
if platform.system() == "Windows":
    DEFAULT_LOG_DIR = os.path.join(os.environ.get("USERPROFILE", "C:\\Users"), "AppData", "Local", "Logs", "med_monitoring")
else:
    DEFAULT_LOG_DIR = "/tmp/app/work/logs/bypass"

LOG_DIR = Path(os.getenv("COZE_LOG_DIR", DEFAULT_LOG_DIR))
