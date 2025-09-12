#!/usr/bin/env python3
"""
Startup script for Orchestrator Agent
Handles initialization, health checks, and graceful startup
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn', 
        'langchain',
        'qdrant_client',
        'google.generativeai'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {missing}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_environment():
    """Check required environment variables"""
    required_vars = ['GEMINI_API_KEY']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Please set them in .env file or environment")
        return False
    
    print("✅ Environment variables are set")
    return True

def check_qdrant_connection():
    """Check if Qdrant is accessible"""
    try:
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()
        print(f"✅ Qdrant is accessible at {host}:{port}")
        return True
        
    except Exception as e:
        print(f"⚠️  Qdrant connection failed: {e}")
        print("Agent will start but RAG functionality may be limited")
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main startup function"""
    print("🚀 Starting Orchestrator Agent")
    print("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Pre-flight checks
    print("🔍 Running pre-flight checks...")
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    # Optional Qdrant check (non-blocking)
    check_qdrant_connection()
    
    print("\n✅ Pre-flight checks completed")
    print("🌟 Starting Orchestrator Agent on port 7000...")
    print("📚 API Documentation: http://localhost:7000/docs")
    print("❤️  Health Check: http://localhost:7000/health")
    print("\nPress Ctrl+C to stop\n")
    
    # Start the FastAPI application
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7000,
            log_level="info",
            access_log=True,
            reload=False  # Set to True for development
        )
        
    except ImportError:
        print("❌ Failed to import uvicorn or main app")
        print("Please ensure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
