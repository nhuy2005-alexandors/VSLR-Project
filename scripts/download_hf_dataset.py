"""Script tự động tải video test từ Máy chủ AWS Singapore & Hugging Face Dataset về thư mục videos/ trên máy tính."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download


def get_active_aws_url() -> str:
    """Lấy địa chỉ Cloudflare Tunnel đang hoạt động của máy chủ AWS từ Auto-Discovery."""
    try:
        req = urllib.request.Request("https://ntfy.sh/vslr_ctu_aws_active_backend_prod_v3/json?poll=1&since=24h")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            lines = resp.read().decode("utf-8").strip().splitlines()
            latest_time = -1
            latest_url = ""
            for line in lines:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if (
                        data.get("event") == "message"
                        and str(data.get("message", "")).startswith("https://")
                        and ".trycloudflare.com" in str(data.get("message", ""))
                        and int(data.get("time", 0)) >= latest_time
                    ):
                        latest_time = int(data.get("time", 0))
                        latest_url = str(data.get("message", "")).strip().rstrip("/")
                except Exception:
                    pass
            return latest_url
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dữ liệu video từ AWS / Hugging Face Dataset")
    parser.add_argument("--token", default="", help="Hugging Face Token")
    parser.add_argument("--dataset-repo", default="ntbii305/vslr-remote", help="Dataset Repo ID")
    parser.add_argument("--signer", default="", help="Chỉ tải thư mục của 1 người cụ thể (VD: khuong, huy)")
    parser.add_argument("--output-dir", default="videos", help="Thư mục lưu trên máy (Mặc định: videos)")
    parser.add_argument("--backend-url", default="", help="Link Cloudflare AWS trực tiếp (nếu muốn chỉ định)")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    env_file = root_dir / "vslr.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    out_dir = root_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải trực tiếp từ Máy chủ AWS Singapore (Qua Auto-Discovery)
    aws_url = (args.backend_url or get_active_aws_url()).strip().rstrip("/")
    if aws_url:
        print(f"\n[1/2] Đang kết nối máy chủ AWS ({aws_url})...")
        signers_to_fetch: list[str] = []
        if args.signer:
            signers_to_fetch = [args.signer.strip()]
        else:
            try:
                list_req = urllib.request.Request(f"{aws_url}/api/recordings/list")
                with urllib.request.urlopen(list_req, timeout=10.0) as resp:
                    list_data = json.loads(resp.read().decode("utf-8"))
                    signers_to_fetch = [s["name"] for s in list_data.get("signers", []) if s.get("name")]
            except Exception:
                signers_to_fetch = [""]

        display_names = ", ".join(signers_to_fetch) if signers_to_fetch else "tất cả"
        print(f"      Tìm thấy người thử nghiệm: {display_names}")

        for s_name in (signers_to_fetch or [""]):
            q_signer = urllib.parse.quote(s_name) if s_name else ""
            zip_endpoint = f"{aws_url}/api/recordings/zip?signer={q_signer}"
            try:
                req = urllib.request.Request(zip_endpoint)
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    zip_bytes = resp.read()
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                        zf.extractall(out_dir)
                    print(f"      ✓ Đã tải và giải nén thành công: '{s_name or 'toàn bộ'}' -> {out_dir}")
            except Exception as exc:
                print(f"      [Thông báo tải '{s_name}'] {exc}")

    # 2. Đồng bộ thêm từ Hugging Face Dataset (nếu có)
    token = (args.token or os.environ.get("HF_TOKEN") or "").strip()
    if token:
        allow_patterns = f"data/*{args.signer}*/*" if args.signer else "data/**"
        print(f"\n[2/2] Đang kiểm tra kho Hugging Face Dataset '{args.dataset_repo}'...")
        try:
            snapshot_download(
                repo_id=args.dataset_repo,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                local_dir=str(out_dir),
                token=token,
            )
        except Exception:
            pass

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
