"""临时：开代理后再测一次 HF 直连速度，并打印当前生效的代理设置。"""
import json
import os
import time
import urllib.request

REPO = "XuShihao6715/counseling-cpsdd"
CHUNK = 20 * 1024 * 1024


def show_proxies():
    print("环境变量代理:")
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
            "https_proxy", "all_proxy", "NO_PROXY", "no_proxy")
    found = False
    for k in keys:
        v = os.environ.get(k)
        if v:
            print(f"  {k} = {v}")
            found = True
    if not found:
        print("  (无)")
    print("getproxies() 探测:")
    try:
        print(f"  {urllib.request.getproxies()}")
    except Exception as e:
        print(f"  失败 {e}")


def test(endpoint, label, use_proxy=True):
    opener = None
    if not use_proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"{endpoint}/api/datasets/{REPO}/tree/main"
    try:
        t0 = time.time()
        with (opener.open(url, timeout=25) if opener
              else urllib.request.urlopen(url, timeout=25)) as r:
            items = json.load(r)
        print(f"  [{label}] API OK  {(time.time()-t0)*1000:.0f}ms  "
              f"{len(items)} 个文件")
    except Exception as e:
        print(f"  [{label}] API 失败: {type(e).__name__}: {str(e)[:90]}")
        return 0.0

    files = [x for x in items if x.get("type") == "file"
             and int(x.get("size") or 0) > CHUNK]
    if not files:
        print("  没有够大的文件可测")
        return 0.0
    files.sort(key=lambda x: -int(x["size"]))
    path = files[0]["path"]
    url = f"{endpoint}/datasets/{REPO}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={
        "Range": f"bytes=0-{CHUNK-1}", "User-Agent": "dreamy-speedtest/1.0"})
    t0 = time.time()
    try:
        with (opener.open(req, timeout=60) if opener
              else urllib.request.urlopen(req, timeout=60)) as r:
            got = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                got += len(b)
        dt = time.time() - t0
        mbps = got / 2**20 / dt
        print(f"  [{label}] {got/2**20:.0f}MB / {dt:.1f}s = {mbps:.2f} MB/s"
              f"  -> 4.98GB 约 {4.98*1024/mbps/60:.0f} 分钟")
        return mbps
    except Exception as e:
        print(f"  [{label}] 下载失败: {type(e).__name__}: {str(e)[:90]}")
        return 0.0


show_proxies()
print("\n[1] huggingface.co 直连（走当前代理设置）")
test("https://huggingface.co", "official")
print("\n[2] huggingface.co 直连（强制不走代理）")
test("https://huggingface.co", "official-noproxy", use_proxy=False)
