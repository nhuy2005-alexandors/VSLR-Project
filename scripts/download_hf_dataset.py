"""Script tải video và dữ liệu test từ Hugging Face Dataset (ntbii305/vslr-remote) về máy tính."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dữ liệu video từ Hugging Face Dataset")
    parser.add_argument("--token", default="", help="Hugging Face Token")
    parser.add_argument("--dataset-repo", default="ntbii305/vslr-remote", help="Dataset Repo ID")
    parser.add_argument("--signer", default="", help="Chỉ tải thư mục của 1 người cụ thể (VD: khach, huy)")
    parser.add_argument("--output-dir", default="videos_from_huggingface", help="Thư mục lưu trên máy")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    env_file = root_dir / "vslr.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    token = (args.token or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print("=" * 68)
        print("   TẢI VIDEO TEST TỪ HUGGING FACE DATASET VỀ MÁY TÍNH")
        print("=" * 68)
        token = input("👉 Nhập mã Hugging Face Token (hf_...): ").strip()

    out_dir = root_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = f"data/{args.signer}/*" if args.signer else "data/**"
    print(f"\n[1/2] Đang tải dữ liệu từ '{args.dataset_repo}' ({allow_patterns})...")

    local_path = snapshot_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(out_dir),
        token=token if token else None,
    )

    # Thống kê và tóm tắt các file JSON vừa tải về
    json_files = sorted(out_dir.rglob("*.json"))
    mp4_files = sorted(out_dir.rglob("*.mp4"))

    print(f"\n[2/2] Hoàn tất! Đã tải về {len(mp4_files)} video (.mp4) và {len(json_files)} bản ghi (.json)")
    print(f"📂 Đường dẫn thư mục trên máy: {out_dir}\n")

    if json_files:
        print("=" * 80)
        print(f"{'THỜI GIAN':<20} | {'NGƯỜI TEST':<12} | {'TRẠNG THÁI':<10} | {'NHÃN DỰ ĐOÁN':<24} | {'CONF':<6} | {'THỜI LƯỢNG'}")
        print("-" * 80)
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                ts = data.get("timestamp", "")[:19].replace("T", " ")
                signer = data.get("signer", "khach")
                status = data.get("status", "")
                label = data.get("label", "")
                conf = f"{data.get('confidence', 0) * 100:.0f}%"
                dur = f"{data.get('duration_seconds', 0):.2f}s ({data.get('total_frames', 0)}f @ {data.get('fps', 0):.1f}fps)"
                print(f"{ts:<20} | {signer:<12} | {status:<10} | {label:<24} | {conf:<6} | {dur}")
            except Exception:
                pass
        print("=" * 80)


if __name__ == "__main__":
    main()
