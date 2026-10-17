"""Minimal local Google Drive downloader.

Replaces the pip ``gdown`` package. Exposes the same ``download(id, output, quiet)``
signature that ``app.py`` and ``webui.py`` use.

Usage::

    from tools.gdown import download
    download(id='1aBcDe...', output='model.zip', quiet=True)

Handles the confirmation-token flow that Google Drive requires for files
larger than ~100 MB. Falls back to a direct fetch for smaller files.
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import tempfile
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


def _build_opener() -> urllib.request.OpenerDirector:
    """A urllib opener that follows redirects and persists cookies."""
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar()),
        urllib.request.HTTPRedirectHandler(),
    )


def _get_confirm_token(contents: str) -> str | None:
    """Extract the confirm-token from the interstitial HTML page."""
    match = re.search(r'confirm=([0-9A-Za-z_]+)', contents)
    return match.group(1) if match else None


def download(id: str, output: str | None = None, quiet: bool = False) -> str | None:
    """Download a Google Drive file by its ``id``.

    Parameters
    ----------
    id : str
        The Google Drive file id (everything after ``/d/`` in the share URL,
        or the ``id=`` query parameter).
    output : str, optional
        Destination path. Defaults to ``./<id>``.
    quiet : bool
        If True, suppress progress messages.

    Returns
    -------
    str or None
        The output path on success, or None on failure.
    """
    if not id:
        raise ValueError('Google Drive file id is required.')

    if output is None:
        output = os.path.join('.', id)
    elif os.path.isdir(output):
        output = os.path.join(output, id)

    opener = _build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (compatible; CoverGen/1.0)')]

    # 1) Initial request — Google may return the file directly, an
    #    interstitial "virus scan" page for large files, or a "quota
    #    exceeded" page.
    base_url = f'https://drive.google.com/uc?id={urllib.parse.quote(id)}'
    if not quiet:
        print(f'[gdown] Downloading {id} -> {output}', file=sys.stderr)

    response = opener.open(base_url)
    content_type = response.headers.get('Content-Type', '')
    # Read the body once. For large files we need to inspect HTML first,
    # so buffer up to 1 MiB; if it's binary we rewind via a temp file.
    body = response.read()

    # 2) If we got HTML back, it's the confirmation page — extract the
    #    confirm token and re-request with it appended.
    if 'text/html' in content_type:
        html = body.decode('utf-8', errors='ignore')
        token = _get_confirm_token(html)
        if token:
            confirm_url = f'{base_url}&confirm={token}'
            if not quiet:
                print('[gdown] Following Google Drive confirmation token.', file=sys.stderr)
            response = opener.open(confirm_url)
            body = response.read()
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                # Quota exceeded or permission denied.
                msg = _extract_error_message(body.decode('utf-8', errors='ignore'))
                raise RuntimeError(f'Google Drive download failed: {msg or "unknown error"}')
        else:
            msg = _extract_error_message(html)
            raise RuntimeError(f'Google Drive download failed: {msg or "could not get confirm token"}')

    # 3) Stream the binary body to the output file.
    os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
    with open(output, 'wb') as f:
        f.write(body)

    if not quiet:
        size = os.path.getsize(output)
        print(f'[gdown] Done. ({size} bytes)', file=sys.stderr)

    return output


def _extract_error_message(html: str) -> str | None:
    """Pull a human-readable error string out of a Google Drive HTML page."""
    for pattern in (
        r'<title>([^<]+)</title>',
        r'class="uc-error-subtitle"[^>]*>\s*([^<]+)\s*<',
        r'<div[^>]*class="[^"]*error[^"]*"[^>]*>\s*([^<]+)\s*<',
    ):
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None
