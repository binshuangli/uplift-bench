"""Download + cache helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(os.environ.get("UPLIFT_DATA_DIR", "data"))


def get_data_dir() -> Path:
    d = _DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path, chunk: int = 1 << 20, ssl_verify: bool = True) -> Path:
    """Download *url* to *dest* if it does not already exist.

    ssl_verify=False disables cert checking for old servers with hostname
    mismatches (e.g. minethatdata.com); only use for known-safe public datasets.
    """
    if dest.exists():
        log.info("Cache hit: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unique staging name per process: concurrent workers on a cold cache must not
    # share one .tmp file (one worker's move makes the other's copy/unlink fail).
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    log.info("Downloading %s -> %s", url, dest)

    ctx = None
    if not ssl_verify:
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        opener = urllib.request.build_opener()
        if ctx is not None:
            opener.add_handler(urllib.request.HTTPSHandler(context=ctx))
        with opener.open(url) as resp, open(tmp, "wb") as fh:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
        shutil.move(str(tmp), str(dest))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def md5(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(buf)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
