"""Download the three public datasets and record SHA-256 checksums.

Sources
- Default of Credit Card Clients (UCI id 350), CC BY 4.0
- Bank Marketing, bank-additional-full (UCI id 222), CC BY 4.0
- Credit Card Fraud Detection (ULB) mirrored on OpenML as dataset 1597 ("creditcard").
  The Kaggle page requires a login; the OpenML mirror is the same 284,807 x 31 table.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "credit_default": {
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "kind": "zip",
        "want": "default of credit card clients.xls",
        "out": "default_of_credit_card_clients.xls",
    },
    "bank_marketing": {
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "kind": "nested-zip",
        "inner_zip": "bank-additional.zip",
        "want": "bank-additional/bank-additional-full.csv",
        "out": "bank-additional-full.csv",
    },
    "fraud": {
        "url": "https://api.openml.org/data/v1/download/1673544/creditcard.arff",
        "kind": "file",
        "out": "creditcard.arff",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nfi-download/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main() -> None:
    manifest = {}
    for name, spec in SOURCES.items():
        out = RAW / spec["out"]
        if out.exists():
            print(f"[skip] {name}: {out.name} exists")
        else:
            print(f"[get ] {name}: {spec['url']}")
            blob = fetch(spec["url"])
            if spec["kind"] == "file":
                out.write_bytes(blob)
            elif spec["kind"] == "zip":
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    out.write_bytes(z.read(spec["want"]))
            elif spec["kind"] == "nested-zip":
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    inner = z.read(spec["inner_zip"])
                with zipfile.ZipFile(io.BytesIO(inner)) as z2:
                    out.write_bytes(z2.read(spec["want"]))
        manifest[name] = {"file": out.name, "sha256": sha256(out), "bytes": out.stat().st_size,
                          "source": spec["url"]}
        print(f"       {out.name}  {manifest[name]['bytes']:,} bytes  sha256={manifest[name]['sha256'][:16]}...")
    (RAW.parent / "checksums.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", RAW.parent / "checksums.json")


if __name__ == "__main__":
    sys.exit(main())
