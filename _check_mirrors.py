"""列出镜像里可用的 torch 版本与具体 wheel 文件名。

pip 的版本号比较会对本地版本标签（+cu130）做特殊处理，
所以「pip 说没有 2.14.1+cu130」不代表镜像里没有 —— 直接看 sed 出来的清单最准。
"""
import html
import re
import urllib.request

CANDIDATES = [
    "https://mirrors.aliyun.com/pytorch-wheels/cu130/",
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/",
    "https://download.pytorch.org/whl/cu130/torch/",
    "https://download.pytorch.org/whl/cu128/torch/",
]

WANT = re.compile(r"torch-(\d+\.\d+\.\d+)\+cu\d+-cp312-cp312-manylinux.*x86_64\.whl")

for base in CANDIDATES:
    host_path = base.replace("https://", "").rstrip("/")
    print(f"\n=== {base}")
    try:
        req = urllib.request.Request(base, headers={"User-Agent": "dreamy/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read().decode("utf-8", "ignore")
    except Exception as e:
        print("  FAIL", type(e).__name__, str(e)[:70])
        continue

    names = [html.unescape(h) for h in re.findall(r"href=\"([^\"]*torch-[^\"]*cp312[^\"]*x86_64\.whl)\"", body)]
    versions = sorted({WANT.search(n).group(1) for n in names if WANT.search(n)})
    print(f"  页面 {len(body) / 1024:.0f}KB，wheel 链接 {len(names)} 个")
    print(f"  版本: {versions}")
    for target in ("2.14.1", "2.15.0"):
        hit = [n for n in names if f"torch-{target}+" in n]
        if hit:
            print(f"  ✓ {target}: {hit[0][:110]}")
