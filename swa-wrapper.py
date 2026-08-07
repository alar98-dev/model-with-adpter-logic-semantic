#!/usr/bin/env python3
"""
SWA - Spatial World Agent CLI
Organic AI System for Hardware-Accelerated Cognition

This script provides a unified interface to the SWA system.
"""

import sys
import os

def main():
    """Main entry point for SWA CLI."""
    # Add current directory to Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # Import and run main SWA application
    try:
        from main import main as swa_main
        swa_main()
    except KeyboardInterrupt:
        print("\n🧬 SWA shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ SWA Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()