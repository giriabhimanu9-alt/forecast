"""ECMWF TIGGE archive route — configuration and access check only.

Findings (verified 2026-09-28):

* The TIGGE Web API (``https://api.ecmwf.int/v1/datasets/tigge``) answers
  ``403 Missing access token`` without credentials. Access needs a free ECMWF
  account, acceptance of the TIGGE licence, and an API key in
  ``~/.ecmwfapirc`` or ``ECMWF_API_URL`` / ``ECMWF_API_KEY`` / ``ECMWF_API_EMAIL``,
  plus ``pip install ecmwf-api-client``.
* TIGGE stores total precipitation (param 228228) at 6-hourly steps only.
  IMD daily windows end at 03 UTC, so from 00/12 UTC cycles no pair of TIGGE
  steps brackets an IMD window exactly (it would need steps 3/27 or 15/39).
  Using TIGGE for IMD verification would require a 3 h time shift or temporal
  interpolation of accumulations, which Phase 2 deliberately does not do.

ECMWF historical verification therefore uses the ECMWF Open Data archive
mirror on AWS (``ecmwf-forecasts`` bucket, 3-hourly steps, no credentials)
through :class:`app.services.data.ecmwf.ECMWFProvider`.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.models.schema import NotConfiguredError

TIGGE_API_URL = "https://api.ecmwf.int/v1"
TIGGE_PRECIP_STEP_HOURS = 6


def tigge_credentials_present() -> bool:
    if all(os.environ.get(k) for k in ("ECMWF_API_URL", "ECMWF_API_KEY", "ECMWF_API_EMAIL")):
        return True
    return (Path.home() / ".ecmwfapirc").is_file()


def tigge_status() -> dict[str, object]:
    """Machine-readable status of the TIGGE route for experiment manifests."""
    return {
        "route": "ECMWF TIGGE (MARS Web API)",
        "credentials_present": tigge_credentials_present(),
        "credential_required": "ECMWF account + TIGGE licence + API key (~/.ecmwfapirc or ECMWF_API_URL/KEY/EMAIL)",
        "precipitation_step_hours": TIGGE_PRECIP_STEP_HOURS,
        "imd_window_compatible": False,
        "used": False,
        "reason": ("6-hourly precipitation steps cannot bracket IMD 03 UTC windows from 00/12 UTC cycles; "
                   "ECMWF Open Data AWS archive (3-hourly) used instead"),
    }


def require_tigge() -> None:
    if not tigge_credentials_present():
        raise NotConfiguredError(
            "TIGGE requires an ECMWF API key: register at ECMWF, accept the TIGGE licence and set "
            "~/.ecmwfapirc or ECMWF_API_URL / ECMWF_API_KEY / ECMWF_API_EMAIL"
        )
    raise NotConfiguredError(
        "TIGGE precipitation is 6-hourly and cannot be aligned exactly with IMD 03 UTC windows; "
        "not used for Phase 2 verification"
    )
