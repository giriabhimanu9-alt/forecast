"""HTTP download helpers with retries, byte ranges, atomic writes and a local cache."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from app.models.schema import DownloadError, InvalidDataError

log = logging.getLogger(__name__)

USER_AGENT = "ps81-forecast-blending/phase1 (python-requests)"


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int | None  # inclusive; None = to end of object

    @property
    def header(self) -> str:
        return f"bytes={self.start}-{'' if self.end is None else self.end}"

    @property
    def expected_length(self) -> int | None:
        return None if self.end is None else self.end - self.start + 1


class HttpClient:
    def __init__(
        self,
        timeout: float = 120.0,
        retries: int = 3,
        session: requests.Session | None = None,
        min_interval: float = 0.0,
        max_backoff: float = 60.0,
    ):
        """``min_interval`` spaces consecutive requests (politeness for bulk
        historical pulls from rate-limited buckets such as the ECMWF S3 mirror,
        which answers ``503 SlowDown``). Retries back off exponentially with
        jitter, capped at ``max_backoff`` seconds."""
        self.timeout = timeout
        self.retries = retries
        self.min_interval = min_interval
        self.max_backoff = max_backoff
        self._last_request = 0.0
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    def _pace(self) -> None:
        if self.min_interval > 0:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._pace()
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                return resp
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    wait = min(self.max_backoff, 2**attempt) * (1.0 + 0.25 * random.random())
                    log.warning("%s %s failed (%s); retrying in %.1fs", method, url, exc, wait)
                    time.sleep(wait)
        raise DownloadError(f"{method} {url} failed after {self.retries + 1} attempts: {last_exc}")

    def exists(self, url: str) -> bool:
        try:
            resp = self._request("HEAD", url, allow_redirects=True)
        except DownloadError:
            return False
        return resp.status_code == 200

    def get_text(self, url: str) -> str:
        resp = self._request("GET", url)
        if resp.status_code != 200:
            raise DownloadError(f"GET {url} returned HTTP {resp.status_code}")
        return resp.text

    def get_bytes(self, url: str, byte_range: ByteRange | None = None) -> bytes:
        headers = {"Range": byte_range.header} if byte_range else {}
        resp = self._request("GET", url, headers=headers)
        expected_status = 206 if byte_range else 200
        if resp.status_code != expected_status:
            raise DownloadError(
                f"GET {url} ({headers.get('Range', 'full')}) returned HTTP {resp.status_code}"
            )
        content = resp.content
        if byte_range and byte_range.expected_length is not None:
            if len(content) != byte_range.expected_length:
                raise DownloadError(
                    f"Truncated download from {url}: got {len(content)} bytes, "
                    f"expected {byte_range.expected_length}"
                )
        return content

    def post_bytes(self, url: str, data: dict[str, str]) -> bytes:
        resp = self._request("POST", url, data=data)
        if resp.status_code != 200:
            raise DownloadError(f"POST {url} {data} returned HTTP {resp.status_code}")
        return resp.content


def write_cached(
    path: Path,
    content: bytes,
    validator: Callable[[Path], None],
    provenance: dict[str, object],
) -> None:
    """Atomically write ``content`` to ``path`` after validating it.

    The file is first written to ``<path>.part``; if ``validator`` raises,
    the partial file is removed and the error propagates, so a corrupted
    download never lands in the cache. A ``<path>.json`` sidecar records
    where the bytes came from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    try:
        validator(tmp)
    except InvalidDataError:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    sidecar = {
        **provenance,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.with_name(path.name + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def cached_file_ok(path: Path, validator: Callable[[Path], None]) -> bool:
    """True if ``path`` exists and passes ``validator``; corrupt cache entries are deleted."""
    if not path.is_file():
        return False
    try:
        validator(path)
    except InvalidDataError as exc:
        log.warning("Discarding corrupt cache file %s: %s", path, exc)
        path.unlink(missing_ok=True)
        return False
    return True
