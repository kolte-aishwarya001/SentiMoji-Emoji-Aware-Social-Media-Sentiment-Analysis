
import re
from urllib.parse import urlparse


def extract_asin(url):
    if not url or not url.strip():
        raise ValueError("Please enter an Amazon product URL.")

    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()

    allowed_hosts = {
        "amazon.in", "www.amazon.in",
        "amazon.com", "www.amazon.com",
        "amazon.co.uk", "www.amazon.co.uk",
    }

    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://.")

    if host not in allowed_hosts:
        raise ValueError("Please enter a supported Amazon URL.")

    match = re.search(
        r"/(?:dp|gp/product|gp/aw/d)/([A-Z0-9]{10})(?=/|$)",
        parsed.path,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    raise ValueError("Product ASIN not found in the URL.")
