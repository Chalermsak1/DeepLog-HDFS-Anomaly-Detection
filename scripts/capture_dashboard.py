"""Capture real screenshots of the running Streamlit cockpit via Chrome DevTools Protocol.

Prerequisites:
    1. streamlit run app.py              (serving on http://localhost:8501)
    2. Chrome started with remote debugging on port 9222
"""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"

TABS = [
    ("SOTA Benchmarks", "cockpit_tab1_benchmarks.png"),
    ("OTel Multi-Modal", "cockpit_tab2_otel.png"),
    ("Chaos Gym", "cockpit_tab3_chaos.png"),
    ("Provable Safety", "cockpit_tab4_safety.png"),
    ("Neuro-Symbolic AST", "cockpit_tab5_ast.png"),
    ("Federated ZK", "cockpit_tab6_mesh.png"),
]


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.i = 0

    async def call(self, method: str, **params):
        self.i += 1
        my_id = self.i
        await self.ws.send(json.dumps({"id": my_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my_id:
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("result", {}).get("value")

    async def shot(self, path: Path):
        r = await self.call("Page.captureScreenshot", format="png")
        data = base64.b64decode(r["data"])
        path.write_bytes(data)
        print(f"Saved {path.name} ({len(data):,} bytes)")


async def main() -> None:
    targets = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json"))
    page = next(t for t in targets if t["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
        cdp = CDP(ws)
        await cdp.call("Page.enable")
        await cdp.call(
            "Emulation.setDeviceMetricsOverride",
            width=1400,
            height=1200,
            deviceScaleFactor=1.5,
            mobile=False,
        )

        for label, fname in TABS:
            clicked = await cdp.js(
                f"""(() => {{
                const tabs = [...document.querySelectorAll('div[data-testid="stTab"][role="tab"]')];
                const target = tabs.find(t => t.innerText && t.innerText.includes({json.dumps(label)}));
                if (!target) return false;
                target.click();
                return true;
            }})()"""
            )
            if not clicked:
                print("Tab not found:", label)
                continue
            await asyncio.sleep(2.5)
            await cdp.js("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
            await cdp.shot(OUT / fname)

    print("All captures finished!")


if __name__ == "__main__":
    asyncio.run(main())
