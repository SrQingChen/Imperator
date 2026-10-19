"""离线预下载模型（与运行时同一代码路径）。用法：
  python scripts/fetch_models.py --list
  python scripts/fetch_models.py --all
  python scripts/fetch_models.py --only mss:model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt sep:MossFormer2_SS_16K
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import events, models_manager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()

    items = models_manager.status()
    if args.list or not (args.all or args.only):
        for it in items:
            mark = "✓" if it["installed"] else "✗"
            print(f"{mark} {it['key']:60s} {it['name']}")
        return 0

    keys = args.only or [it["key"] for it in items if not it["installed"]]
    for key in keys:
        print(f"==> 下载 {key}")
        ok = models_manager.start_download(key)
        if not ok:
            print("    已在下载中/未受理")
            continue
        while True:
            st = next((m.get("download", {}) for m in models_manager.status() if m["key"] == key), {})
            if st.get("status") in ("done", "error"):
                print(f"    {st['status']}: {st.get('error', '') or '完成'}")
                break
            print(f"    {st.get('done_mb', 0)} MB / {st.get('total_mb', '?')} MB", end="\r")
            time.sleep(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
