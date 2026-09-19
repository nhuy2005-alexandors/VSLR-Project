"""Seal P05 final evaluation artifacts with PROVENANCE_MANIFEST.json and SHA256SUMS.txt."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

eval_dir = Path("evaluation/p05-v3-final-20260916")

# 1. Write PROVENANCE_MANIFEST.json
prov = {
    "evaluation_id": "p05-v3-final-20260916",
    "executed_at_utc": "2026-09-16T00:49:19.5200192Z",
    "P05_OPENED": True,
    "candidate_checkpoint_sha256": "e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4",
    "candidate_run_dir": "runs/v3-ship-4signers-24-8clips-13ep-20260916-020145",
    "candidate_run_manifest_sha256": "6e1c624bfc164fe2fb7b560e558e46b58542d8b6dbf510658e9991faf6889909",
    "candidate_training_manifest_sha256": "9c299c37e793fffb55932b96b501f047c82a29d0d65a8984f4357c838796fd57",
    "candidate_training_signature": "918588dedeb2e9bdb2dabf8a2f8ac843aa6cf66dc68d61c2e8acb1e4b07a65cf",
    "git_head": "88db231be73687c7a8a421f25de0f7add79da018",
    "branch": "pipeline-signer-split",
    "test_set_fingerprint": "0ac379cb4185e46b0cf24215e120b644458fd6aceda8fd03008b4648932a5bc8",
    "evaluation_summary": {
        "total_clips": 48,
        "raw_top1_accuracy": 1.0,
        "top3_accuracy": 1.0,
        "coverage": 1.0,
        "rejection_rate": 0.0,
        "accepted_accuracy": 1.0,
        "reject_policy_calibrated": False,
        "confidence_threshold": 0.50
    }
}

prov_path = eval_dir / "PROVENANCE_MANIFEST.json"
prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {prov_path}")

# 2. Generate recursive SHA256SUMS.txt (UTF-8 no BOM)
lines = []
for p in sorted(eval_dir.rglob("*")):
    if p.is_file() and p.name != "SHA256SUMS.txt":
        sha = hashlib.sha256(p.read_bytes()).hexdigest().lower()
        rel = p.relative_to(eval_dir).as_posix()
        lines.append(f"{sha}  {rel}")

sums_path = eval_dir / "SHA256SUMS.txt"
sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {sums_path} ({len(lines)} files)")

# 3. Verify checksums immediately
check_lines = sums_path.read_text(encoding="utf-8").splitlines()
errors = []
for line in check_lines:
    if not line.strip(): continue
    exp_sha, rel = line.split(maxsplit=1)
    target = eval_dir / rel.strip()
    if not target.is_file():
        errors.append(f"Missing file: {rel}")
    else:
        act_sha = hashlib.sha256(target.read_bytes()).hexdigest().lower()
        if act_sha != exp_sha.lower():
            errors.append(f"Checksum mismatch in {rel}: {act_sha} != {exp_sha}")

assert len(errors) == 0, f"Checksum verification errors: {errors}"
print(f"Checksum verification PASS: {len(check_lines)} files verified with 0 mismatches!")
