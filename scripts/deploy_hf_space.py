"""Script tự động đẩy Backend VSLR và cấu hình Secrets lên Hugging Face Space (ntbii305/vslr-backend)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Đẩy Backend VSLR lên Hugging Face Space")
    parser.add_argument("--token", default="", help="Hugging Face Access Token (Write)")
    parser.add_argument("--space-repo", default="ntbii305/vslr-backend", help="Repo ID của Space")
    parser.add_argument("--dataset-repo", default="ntbii305/vslr-remote", help="Repo ID của Dataset lưu video")
    parser.add_argument("--sync-secrets", action="store_true", default=False, help="Đồng bộ lại Secrets")
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
        print("   CẤU HÌNH & ĐẨY BACKEND LÊN HUGGING FACE SPACE (TỰ ĐỘNG 100%)")
        print("=" * 68)
        token = input("👉 Nhập mã Hugging Face Token (hf_... có quyền WRITE): ").strip()

    if not token or not token.startswith("hf_"):
        print("[LỖI] Mã Token không hợp lệ! Vui lòng nhập mã bắt đầu bằng 'hf_...'", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    if args.sync_secrets:
        print(f"\n[1/3] Đang xử lý Variables/Secrets cho Space '{args.space_repo}'...")
        for key_to_clean in ("HF_TOKEN", "HF_DATASET_REPO"):
            try:
                api.delete_space_variable(repo_id=args.space_repo, key=key_to_clean)
            except Exception:
                pass
        try:
            api.add_space_secret(repo_id=args.space_repo, key="HF_TOKEN", value=token)
            api.add_space_secret(repo_id=args.space_repo, key="HF_DATASET_REPO", value=args.dataset_repo)
            print("      ✓ Đã cài đặt thành công HF_TOKEN, HF_DATASET_REPO!")
        except Exception as exc:
            print(f"      [Cảnh báo] {exc}")
    else:
        print(f"\n[1/3] Bỏ qua cập nhật Secrets (đã có sẵn) để đẩy code nhanh và không khởi động lại 2 lần.")

    print(f"\n[2/3] Đang kiểm tra kho lưu trữ Dataset '{args.dataset_repo}'...")
    try:
        api.create_repo(repo_id=args.dataset_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"      ✓ Kho Dataset '{args.dataset_repo}' đã sẵn sàng!")
    except Exception as exc:
        print(f"      [Thông báo] Dataset '{args.dataset_repo}': {exc}")

    print(f"\n[3/3] Đang tải mã nguồn AI & Mô hình BiLSTM lên Space '{args.space_repo}'...")

    allow_patterns = [
        "app.py",
        "packages.txt",
        "requirements.txt",
        "models/**",
        "artifacts/v3-realtime-test-candidate/gesture_lstm.pt",
        "artifacts/v3-realtime-test-candidate/labels.json",
        "artifacts/v3-realtime-test-candidate/metrics.json",
        "src/**",
        "vslr-web/*.html",
        "vslr-web/*.js",
        "vslr-web/*.css",
        "vslr-web/*.png",
        "vslr-web/vercel.json",
    ]
    ignore_patterns = [
        "**/__pycache__/**",
        "**/*.pyc",
        "venv/**",
        "dataset/**",
        "videos/**",
        ".git/**",
        ".pytest_cache/**",
    ]

    commit_info = api.upload_folder(
        repo_id=args.space_repo,
        repo_type="space",
        folder_path=str(root_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message="Deploy VSLR Realtime Backend & Hugging Face Dataset Sync",
    )

    print("\n" + "=" * 68)
    print("🎉 ĐÃ ĐẨY THÀNH CÔNG LÊN HUGGING FACE SPACE!")
    print(f"🔗 Trang quản lý Space: https://huggingface.co/spaces/{args.space_repo}")
    print(f"🔗 Địa chỉ API Backend: https://ntbii305-vslr-backend.hf.space")
    print(f"🔗 Kho lưu trữ Video:   https://huggingface.co/datasets/{args.dataset_repo}")
    print("=" * 68)


if __name__ == "__main__":
    main()
