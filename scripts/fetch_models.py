"""
Download the CatGAN-26 release assets (too large for git) into the paths the scripts expect, and
check their SHA-256 against MODELS.md. Already-present files are only checked.

  python scripts/fetch_models.py                       weights only (needed for exp12-exp15, cat git demos)
  python scripts/fetch_models.py --with-checkpoint     also the 217 MB training checkpoint (exp11 resume, exp17)
  python scripts/fetch_models.py --base-url URL        another mirror (e.g. the Zenodo record's file URL prefix)
"""
import argparse, hashlib, os, sys, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE_URL = "https://github.com/v-code-895/wsse/releases/download/v1.0.0/"     # assets attached to the v1.0.0 release
ASSETS = {   # release file name -> (path in the repository, sha256, needed by default)
    "catgan26.npz": ("results/exp11_out/catgan26.npz", "3aa01a82bffc6a82f3ec1f007625bb8fe67602386347124a406b878b7fa9b23a", True),
    "catgan26_with_ops.npz": ("results/exp13_out/catgan26_with_ops.npz", "abf82f3dbaffc7cbabd70440413e7ea10a3ff4adb49aa55d1eeff62b3ffb73ae", True),
    "catgan26.pt": ("results/exp11_out/catgan26.pt", "b1fb0182295916086e9a2a0f8572a94e4cd9dec449df8d329f79ec03b7815bc6", False),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base-url", default=BASE_URL); ap.add_argument("--with-checkpoint", action="store_true")
    a = ap.parse_args(); ok = True
    for name, (rel, digest, default) in ASSETS.items():
        if not (default or a.with_checkpoint): continue
        path = os.path.normpath(os.path.join(ROOT, rel)); os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            print(f"downloading {name} -> {rel}")
            urllib.request.urlretrieve(a.base_url.rstrip("/") + "/" + name, path)
        good = sha256(path) == digest; ok &= good
        print(f"{'ok      ' if good else 'MISMATCH'} {rel}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
