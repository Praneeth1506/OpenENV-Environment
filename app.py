import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'demo'))

from demo_app import app

app.launch()