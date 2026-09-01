import os
import sys
import subprocess
import time
import signal

def main():
    print("="*60)
    print("AI-QTriage - Process Runner")
    print("="*60)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Start FastAPI Backend
    print("[1/2] Launching FastAPI Backend...")
    backend_cmd = [
        os.path.join(root_dir, "backend", "venv", "Scripts", "python.exe"),
        "-m", "uvicorn", 
        "backend.main:app", 
        "--host", "127.0.0.1", 
        "--port", "8000"
    ]
    
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Give the backend a couple seconds to check DB connection
    time.sleep(2)
    
    # Check if backend exited early (e.g. DB connection failure)
    backend_proc.poll()
    if backend_proc.returncode is not None:
        print("ERROR: FastAPI Backend failed to start. Console output:")
        print(backend_proc.stdout.read())
        sys.exit(1)
    else:
        print("      Success! Backend listening on http://127.0.0.1:8000")

    # 2. Start Next.js Frontend
    print("[2/2] Launching Next.js Frontend...")
    # On Windows, npm is npm.cmd
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev", "--", "--port", "3000"]
    
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=os.path.join(root_dir, "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    time.sleep(2)
    frontend_proc.poll()
    if frontend_proc.returncode is not None:
        print("ERROR: Next.js Frontend failed to start. Console output:")
        print(frontend_proc.stdout.read())
        backend_proc.terminate()
        sys.exit(1)
    else:
        print("      Success! Frontend listening on http://127.0.0.1:3000")

    print("\nBoth services are running in the background.")
    print("Press Ctrl+C to terminate both servers.")
    print("="*60 + "\n")
    
    # Print live logs from both processes
    try:
        while True:
            # Check backend output
            b_line = backend_proc.stdout.readline()
            if b_line:
                print(f"[Backend] {b_line.strip()}")
                
            # Check frontend output
            f_line = frontend_proc.stdout.readline()
            if f_line:
                print(f"[Frontend] {f_line.strip()}")
                
            # Sleep briefly to reduce CPU usage
            time.sleep(0.1)
            
            # Check if any process terminated
            if backend_proc.poll() is not None:
                print("Backend terminated.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend terminated.")
                break
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        # Clean up processes on exit
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=2)
        except Exception:
            pass
        try:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=2)
        except Exception:
            pass
        print("Servers shutdown complete.")

if __name__ == "__main__":
    main()
