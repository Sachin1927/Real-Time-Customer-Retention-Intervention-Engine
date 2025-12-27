import subprocess
import time
import sys
import os

def run():
    print(" Project Sentinel is starting...")

    # 1. Start the Backend (FastAPI)
    # We use sys.executable to ensure we use the same Python environment (venv)
    print(" Starting Backend API...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app", "--reload", "--port", "8000"],
        env=os.environ.copy()
    )

    # Wait 3 seconds for the API to boot up before launching the dashboard
    time.sleep(3)

    # 2. Start the Frontend (Streamlit)
    print(" Starting Frontend Dashboard...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/dashboard/frontend.py"],
        env=os.environ.copy()
    )

    print("\n System Online!")
    print("    Backend: http://127.0.0.1:8000/docs")
    print("    Frontend: http://localhost:8501")
    print("\nPress Ctrl+C to stop both servers.")

    try:
        # Keep the script running to monitor the processes
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        # If user presses Ctrl+C, kill both processes
        print("\n🛑 Stopping services...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("👋 Shutdown complete.")

if __name__ == "__main__":
    run()