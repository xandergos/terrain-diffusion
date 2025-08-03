#!/usr/bin/env python3
"""
Infinite Terrain Explorer Web App

This script starts a web application for exploring infinite terrain generated
by the terrain diffusion model. The terrain is generated on-demand as you 
navigate around.

Usage:
    python run_terrain_viewer.py

Then open your browser to: http://localhost:5000
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terrain_diffusion.web_app.unbdd_pipe_consistency_viewer import app

if __name__ == '__main__':
    print("🏔️  Starting Infinite Terrain Explorer...")
    print("📍 Open your browser to: http://localhost:5000")
    print("⌨️  Use WASD or arrow keys to navigate")
    print("🖱️  Click and drag to pan around")
    print("🔍 Hover over terrain to see elevation")
    print("---")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1) 