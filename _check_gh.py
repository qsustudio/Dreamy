"""测试各 GitHub 镜像对 raw 文件的可达性。"""
import time
import urllib.request

PATH = "qsustudio/Dreamy/main/scripts/train_peft.py"
URLS = [
    f"https://raw.githubusercontent.com/{PATH}",
    f"https://ghproxy.net/https://raw.githubusercontent.com/{PATH}",
    f"https://gh-proxy.com/https://raw.githubusercontent.com/{PATH}",
    f"https://ghfast.top/https://raw.githubusercontent.com/{PATH}",
    f"https://cdn.jsdelivr.net/gh/qsustudio/Dreamy@main/scripts/train_peft.py",
    f"https://gitclone.com/github.com/{PATH}",
]

for u in URLS:
    host = u.split("/")[2]
    t0 = time.time()
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "dreamy/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read()
        dt = time.time() - t0
        ok = b"def main" in body or b"Dreamy" in body
        print(f"  {r.status} {len(body):>8,}B {dt:5.1f}s  {host:<38} 内容OK={ok}")
    except Exception as e:
        print(f"  FAIL     {time.time() - t0:5.1f}s  {host:<38} {type(e).__name__}: {str(e)[:50]}")
