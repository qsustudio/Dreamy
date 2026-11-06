# setup.ps1 -- one-shot, reproducible deployment of x-transformers on Windows
#
# What it does:
#   1. clones lucidrains/x-transformers into .\x-transformers
#   2. creates an isolated venv at .\.venv
#   3. bootstraps pip (venv's ensurepip is blocked by the sandbox DACL)
#   4. installs CPU-only PyTorch + x-transformers (editable) + test deps
#
# The repo is installed in editable mode, so `git pull` inside .\x-transformers
# takes effect immediately without reinstalling.
#
# Usage:
#   pwsh -File _deploy\setup.ps1
#   pwsh -File _deploy\setup.ps1 -TorchVersion 2.15.0+cpu   # pin a torch build
#
# Notes for this machine:
#   * git/curl use schannel TLS here and fail with SEC_E_NO_CREDENTIALS.
#     We force the OpenSSL backend for git, and download wheels with Python's
#     urllib instead of curl.
#   * tempfile.mkdtemp() produces unreadable directories under the sandbox, so
#     all pip work goes through _deploy\pip_wrapper.py with TEMP pointed at
#     .\.venv\pip-tmp.

[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    # 'cpu', 'cu126', 'cu128', 'cu130' ... -> torch build to install
    # (T4 is Turing/sm_75: any cu126+ CUDA build supports it)
    [string]$Cuda = "cpu",
    [string]$TorchVersion = "",
    [string]$RepoUrl = "https://github.com/lucidrains/x-transformers.git",
    [switch]$SkipTests
)

# derived names ------------------------------------------------
# 'cpu' builds are versioned "2.15.0+cpu"; CUDA needs the matching wheel index.
$IsCuda = $Cuda -ne "cpu"
$IndexUrl = "https://download.pytorch.org/whl/$Cuda"
$LocalTag = $Cuda

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoDir = Join-Path $Root "x-transformers"
$VenvDir = Join-Path $Root ".venv"
$VenvPy  = Join-Path $VenvDir "Scripts\python.exe"
$PipTmp  = Join-Path $VenvDir "pip-tmp"
$Wrapper = Join-Path $PSScriptRoot "pip_wrapper.py"
$WheelsDir = Join-Path $Root ".wheels"

function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. clone
Step "Clone x-transformers"
if (Test-Path (Join-Path $RepoDir ".git")) {
    Write-Host "already cloned, skipping"
} else {
    # schannel cannot acquire credentials here -> force the openssl backend
    & git -c http.sslBackend=openssl clone --depth 1 $RepoUrl $RepoDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

# ---------------------------------------------------------------- 2. venv
Step "Create virtual environment"
if (-not (Test-Path $VenvPy)) {
    # use the system interpreter that produced this script's environment
    $sysPy = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $sysPy) { throw "no 'python' on PATH" }
    & $sysPy -m venv $VenvDir
}
New-Item -ItemType Directory -Force -Path $PipTmp, $WheelsDir | Out-Null

# pip work always runs with a writable temp dir
$env:TEMP = $PipTmp
$env:TMP  = $PipTmp

function Invoke-Pip {
    # pass args positionally; -e must reach pip literally, so avoid a named param
    param([string[]]$PipArgs)
    & $VenvPy $Wrapper @PipArgs
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($PipArgs -join ' ')" }
}

# ---------------------------------------------------------------- 3. pip
Step "Bootstrap pip"
& $VenvPy -c "import pip" 2>$null
if ($LASTEXITCODE -ne 0) {
    # venv's ensurepip fails on the sandbox DACL -> extract the wheel by hand
    & $VenvPy -c @"
import json, urllib.request, zipfile, os
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/pip/json', timeout=60))
whl = [f for f in d['urls'] if f['filename'].endswith('.whl')][0]
tmp = os.path.join(r'$($PipTmp)', 'pip.whl')
urllib.request.urlretrieve(whl['url'], tmp)
zipfile.ZipFile(tmp).extractall(os.path.join(r'$($VenvDir)', 'Lib', 'site-packages'))
print('pip', d['info']['version'], 'extracted')
"@
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }
}
& $VenvPy -m pip --version

# ---------------------------------------------------------------- 4. torch
if (-not $TorchVersion) {
    Step "Resolve newest torch build on the $Cuda index"
    $TorchVersion = (& $VenvPy -c @"
import re, urllib.request
req = urllib.request.Request('$IndexUrl/torch/', headers={'User-Agent': 'pip/26.3'})
html = urllib.request.urlopen(req, timeout=60).read().decode()
tag = '+$LocalTag'
vers = {re.search(r'torch-([0-9][^-]*?)%2B', l).group(1)
        for l in re.findall(r'href="([^"]*cp\d+-\w+-win_amd64\.whl[^"]*)"', html)
        if '%2B' in l}
if not vers:
    raise SystemExit('no torch wheel published on $IndexUrl')
print(max(vers, key=lambda v: [int(x) for x in v.split('.') if x.isdigit()]))
"@).Trim()
}
Step "Install PyTorch $TorchVersion (index: $IndexUrl)"

$pyTag = (& $VenvPy -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
$fullVer = "$TorchVersion+$LocalTag"
$torchWheel = Join-Path $WheelsDir "torch-$fullVer-$pyTag-$pyTag-win_amd64.whl"

& $VenvPy -c "import torch, sys; sys.exit(0 if torch.__version__ == '$fullVer' else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (-not (Test-Path $torchWheel)) {
        # fetch the exact wheel with urllib (curl/git TLS is broken on this box)
        & $VenvPy -c @"
import re, os, urllib.request
py, tag = '$pyTag', '$LocalTag'
req = urllib.request.Request('$IndexUrl/torch/', headers={'User-Agent': 'pip/26.3'})
html = urllib.request.urlopen(req, timeout=60).read().decode()
pat = r'href="([^"]*torch-$TorchVersion%%2B%s-%s-%s-win_amd64\.whl[^"]*)"' % (tag, py, py)
m = re.findall(pat, html)
if not m:
    raise SystemExit('no wheel: torch $TorchVersion+%s / %s on $IndexUrl' % (tag, py))
url = m[0].split('#')[0]
# normalise host-relative hrefs to absolute
if url.startswith('/'):
    url = 'https://download.pytorch.org' + url
dest = os.path.join(r'$WheelsDir', os.path.basename(urllib.request.unquote(url)))
req = urllib.request.Request(url, headers={'User-Agent': 'pip/26.3'})
with urllib.request.urlopen(req, timeout=180) as r, open(dest, 'wb') as f:
    while True:
        b = r.read(1 << 20)
        if not b: break
        f.write(b)
print(dest)
"@
    }
    if (-not (Test-Path $torchWheel)) { throw "could not obtain $torchWheel" }

    Invoke-Pip @("install", "--no-cache-dir", "--no-deps", "--force-reinstall", $torchWheel)
    # torch's own runtime deps (skipped above so pip won't pull a different build)
    Invoke-Pip @("install", "--no-cache-dir",
        "sympy>=1.13.3", "filelock", "typing-extensions>=4.13.0",
        "setuptools>=77.0.3", "jinja2", "fsspec>=0.8.5", "networkx")
} else {
    Write-Host "torch $fullVer already installed"
}

# ---------------------------------------------------------------- 5. package
Step "Install x-transformers (editable) + test deps"
Invoke-Pip @("install", "--no-cache-dir", "hatchling", "editables")
Invoke-Pip @("install", "--no-cache-dir", "--no-build-isolation", "-e", $RepoDir)
Invoke-Pip @("install", "--no-cache-dir", "pytest")

# ---------------------------------------------------------------- 6. verify
Step "Smoke test"
& $VenvPy -c @"
import torch, x_transformers as xt, importlib.metadata as md
from x_transformers import TransformerWrapper, Decoder

print('python      :', __import__('sys').version.split()[0])
print('torch       :', torch.__version__)
print('x-transform :', md.version('x-transformers'), '->', xt.__file__)

if torch.cuda.is_available():
    print('cuda        :', torch.version.cuda, '| backend',
          'cu' + torch.version.cuda.replace('.', '') if torch.version.cuda else '')
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = 'sm_%d%d' % cap
        print('gpu %d       :' % i, torch.cuda.get_device_name(i), '|', arch,
              '| supported:', arch in torch.cuda.get_arch_list())
else:
    print('cuda        : no GPU visible (expected on a CPU-only machine)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = TransformerWrapper(num_tokens=256, max_seq_len=64,
                       attn_layers=Decoder(dim=64, depth=2, heads=4)).to(device)
logits = m(torch.randint(0, 256, (2, 32), device=device))
logits.sum().backward()
print('forward     :', tuple(logits.shape), 'on', device)
print('grads       :', sum(p.grad is not None for p in m.parameters()), '/',
                     sum(1 for _ in m.parameters()))
"@
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

if (-not $SkipTests) {
    Step "Test suite"
    & $VenvPy -m pytest (Join-Path $RepoDir "tests") -q -p no:cacheprovider `
        -W ignore::FutureWarning --no-header
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "some tests failed (see _deploy\README.md for known platform limits)"
    }
}

Step "Done"
Write-Host "Activate with:  $VenvDir\Scripts\Activate.ps1"
