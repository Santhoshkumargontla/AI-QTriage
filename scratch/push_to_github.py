import subprocess
import sys
import os

repo_dir = r"c:\Users\santh\Capstone Project Code"

def run_cmd(cmd_list):
    print(">>> Executing:", " ".join(cmd_list), flush=True)
    res = subprocess.run(cmd_list, capture_output=True, text=True, cwd=repo_dir)
    print("STDOUT:", res.stdout.strip(), flush=True)
    print("STDERR:", res.stderr.strip(), flush=True)
    print("RETURN CODE:", res.returncode, flush=True)
    print("=" * 60, flush=True)
    return res.returncode

if __name__ == "__main__":
    lock_file = os.path.join(repo_dir, ".git", "index.lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print("Removed stale lock file.", flush=True)
        except Exception as e:
            print("Lock file error:", e, flush=True)

    print("Staging files...", flush=True)
    run_cmd(["git", "add", "."])
    
    print("Committing files...", flush=True)
    run_cmd(["git", "commit", "-m", "first commit"])
    
    print("Renaming branch to main...", flush=True)
    run_cmd(["git", "branch", "-M", "main"])
    
    print("Configuring remote URL...", flush=True)
    run_cmd(["git", "remote", "remove", "origin"])
    run_cmd(["git", "remote", "add", "origin", "https://github.com/Santhoshkumargontla/AI-QTriage.git"])
    
    print("Pushing to GitHub...", flush=True)
    run_cmd(["git", "push", "-u", "origin", "main"])
