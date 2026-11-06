# Run pip with tempfile.mkdtemp reimplemented via os.makedirs.
#
# Why this exists:
#   On this machine the workspace sits inside a Windows sandbox. Directories
#   created by tempfile.mkdtemp() receive a DACL that denies access even to the
#   owning user -- icacls cannot even read it ("Access is denied"), so pip
#   crashes with:
#       OSError: [Errno 13] Permission denied: '...\pip-unpack-xxxx\...whl'
#   Directories created by os.makedirs() inherit the parent ACL correctly.
#
#   pip, hatchling and setuptools all call tempfile.mkdtemp(), so we substitute
#   an implementation built on os.makedirs() and then run pip normally.
#
# Usage:
#   python _deploy/pip_wrapper.py install <args...>
#   python _deploy/pip_wrapper.py --version
#
# TEMP/TMP should point at a writable directory (the venv\pip-tmp folder that
# setup.ps1 creates), otherwise mkdtemp emulation lands in the blocked temp dir.
import os
import sys
import tempfile
import uuid


def _mkdtemp_makedirs(suffix=None, prefix=None, dir=None):
    base = dir if dir is not None else tempfile.gettempdir()
    prefix = "tmp" if prefix is None else prefix
    suffix = "" if suffix is None else suffix

    for _ in range(100):
        name = os.path.join(base, f"{prefix}{uuid.uuid4().hex[:8]}{suffix}")
        try:
            os.makedirs(name, exist_ok=False)
        except FileExistsError:
            continue
        return name

    raise FileExistsError(f"could not create a unique temp dir under {base}")


# patch the module-level function that TemporaryDirectory and pip both use
tempfile.mkdtemp = _mkdtemp_makedirs


def main(argv):
    from pip._internal.cli.main import main as pip_main

    return pip_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
