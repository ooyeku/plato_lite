import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project storage directory
PROJECT_DIR = os.path.join(ROOT_DIR, 'projects')

# Data directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROJECT_DIR, exist_ok=True)

# Database configuration
DB_NAME = 'plato_lite.db'
DB_PATH = os.path.join(DATA_DIR, DB_NAME)


