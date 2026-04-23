"""Produce a single self-contained HTML so the demo runs from file:// (Finder).

Inlines demo/data.json into the HTML as window.__DEMO_DATA__ and rewrites each
sample's heatmap URL to a data: URI pointing at the base64-encoded PNG.

Usage:
    python3 scripts/bundle_standalone.py
    open demo/standalone.html  # double-click from Finder also works
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "demo"


def main() -> None:
    data = json.loads((DEMO / "data.json").read_text())
    for ds in data["datasets"].values():
        for s in ds["samples"]:
            rel = s.get("heatmap")
            if not rel:
                continue
            png = (DEMO / rel).read_bytes()
            s["heatmap"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    html = (DEMO / "index.html").read_text()
    # Safe inlining: escape closing </script> in the JSON payload so the </script> tag
    # that follows doesn't break out. Replace ``</`` inside the string only.
    inline = (
        "<script>window.__DEMO_DATA__ = "
        + json.dumps(data).replace("</", "<\\/")
        + ";</script>\n</head>"
    )
    html_out = html.replace("</head>", inline, 1)
    out = DEMO / "standalone.html"
    out.write_text(html_out)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
