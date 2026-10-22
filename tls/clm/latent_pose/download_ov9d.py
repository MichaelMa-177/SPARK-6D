"""Download a frozen, bounded OV9D training subset from the authors' host."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile

import requests

BASE = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/ov9d"
# From the authors' prepare_data.py, recorded verbatim as category identifiers.
TEST_CATEGORIES = {"bowl", "bumbag", "dumpling", "facial_cream", "handbag",
                   "litchi", "mouse", "pineapple", "teddy_bear", "toy_truck"}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b""):
            h.update(chunk)
    return h


def download_one(root, name):
    destination = root/"objects"/name.removesuffix(".tgz")
    receipt = root/"receipts"/(name+".json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if destination.exists():
            return saved
        raise ValueError("download receipt exists but extracted directory is missing")
    archive = root/"archives"/name
    url = f"{BASE}/oo3d9dsingle/{name}"
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        expected_md5 = response.headers.get("Content-MD5")
        partial = archive.with_suffix(".tgz.partial")
        with partial.open("wb") as stream:
            for chunk in response.iter_content(4*1024**2):
                stream.write(chunk)
        if expected_md5 and base64.b64encode(digest(partial, "md5").digest()).decode() != expected_md5:
            raise ValueError("author-host Content-MD5 mismatch")
        size = response.headers.get("Content-Length")
        if size and partial.stat().st_size != int(size):
            raise ValueError("archive length mismatch")
        partial.replace(archive)
        metadata = {"archive": name, "source": url, "etag": response.headers.get("ETag"),
                    "content_md5": expected_md5, "bytes": archive.stat().st_size,
                    "sha256": digest(archive).hexdigest()}
    # Extract into one staging directory; refuse links, traversal and collisions.
    staging = root/"staging"/name.removesuffix(".tgz")
    staging.mkdir(parents=True, exist_ok=False)
    members = 0
    with tarfile.open(archive, "r:gz") as tar:
        for item in tar:
            relative = Path(item.name)
            if relative.is_absolute() or ".." in relative.parts or not (item.isfile() or item.isdir()):
                raise ValueError("unsafe training archive member")
            target = staging/relative
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(item) as source, target.open("xb") as stream:
                    shutil.copyfileobj(source, stream)
                members += 1
    staging.replace(destination)
    metadata["extracted_files"] = members
    receipt.write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--categories", type=int, default=64)
    p.add_argument("--instances", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if min(args.categories, args.instances, args.workers) < 1:
        p.error("counts must be positive")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("objects", "archives", "receipts", "staging"):
        (root/directory).mkdir(exist_ok=True)
    policy = root/"policy.json"
    if policy.exists():
        frozen = json.loads(policy.read_text())
        if frozen["selection"] != {k: getattr(args, k) for k in ("categories", "instances", "seed")}:
            raise ValueError("existing frozen selection differs from requested subset")
    else:
        response = requests.get(f"{BASE}/oo3d9dsingle.txt", timeout=60)
        response.raise_for_status()
        content = response.content
        (root/"oo3d9dsingle.txt").write_bytes(content)
        groups = {}
        for name in content.decode().splitlines():
            if Path(name).name != name or not name.endswith(".tgz"):
                raise ValueError("invalid archive name in author list")
            category = "_".join(name.split("_")[:-2])
            if category not in TEST_CATEGORIES:
                groups.setdefault(category, []).append(name)
        eligible = sorted(k for k, v in groups.items() if len(v) >= args.instances)
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(eligible, args.categories))
        names = [name for category in chosen for name in sorted(rng.sample(sorted(groups[category]), args.instances))]
        frozen = {
            "source": "https://github.com/caijunhao/ov9d/blob/main/download_data.py",
            "split_source": "https://github.com/caijunhao/ov9d/blob/main/prepare_data.py",
            "selection": {k: getattr(args, k) for k in ("categories", "instances", "seed")},
            "author_list_sha256": hashlib.sha256(content).hexdigest(),
            "excluded_official_test_categories": sorted(TEST_CATEGORIES),
            "selected_categories": chosen, "archives": names,
            "purpose": "independent synthetic training subset; no YCB-V test images",
        }
        policy.write_text(json.dumps(frozen, indent=2)+"\n")
    print(json.dumps({"event": "policy_frozen", "objects": len(frozen["archives"])}), flush=True)
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, root, name): name for name in frozen["archives"]}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(json.dumps({"done": len(reports), "archive": report["archive"],
                              "bytes": report["bytes"], "files": report["extracted_files"]}), flush=True)
    (root/"complete.json").write_text(json.dumps({
        "objects": len(reports), "archive_bytes": sum(r["bytes"] for r in reports),
        "policy_sha256": digest(policy).hexdigest()}, indent=2)+"\n")


if __name__ == "__main__":
    main()
