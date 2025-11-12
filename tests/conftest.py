import os
import sys

# Ensure project root is on sys.path so 'app_data', 'chat', and 'ml' can be imported in tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
