"""Fetch selected BOP scenes from a pinned official ZIP using HTTP byte ranges."""

import argparse
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import zipfile
import zlib

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, session, url, size, block_size=8*1024**2):
        self.session, self.url, self.size = session, url, size
        self.block_size, self.position, self.downloaded = block_size, 0, 0
        self.cache = OrderedDict()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("negative archive seek")
        self.position = position
        return position

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        count = max(0, min(count, self.size-self.position))
        if count > 128*1024**2:
            raise ValueError("unexpectedly large ZIP read")
        result = bytearray()
        while count:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                end = min(self.size, start + self.block_size) - 1
                for attempt in range(4):
                    try:
                        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                              timeout=(30, 120), stream=True) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                                raise ValueError("server did not honor the requested archive byte range")
                            content = response.content
                            if len(content) != end-start+1:
                                raise ValueError("incomplete archive byte range")
                            break
                    except (requests.RequestException, ValueError):
                        if attempt == 3:
                            raise
                self.downloaded += len(content)
                self.cache[block] = content
                if len(self.cache) > 3:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            within = self.position % self.block_size
            length = min(count, len(self.cache[block])-within)
            if length <= 0:
                raise ValueError("invalid range-reader offset")
            result += self.cache[block][within:within+length]
            count -= length
            self.position += length
        return bytes(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", choices=["ycbv_test_all.zip", "ycbv_test_bop19.zip"],
                   default="ycbv_test_all.zip")
    p.add_argument("--scenes", nargs="+", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--proxy-auth-file", help="optional local proxy credentials; never written into provenance")
    p.add_argument("--include-masks", action="store_true")
    args = p.parse_args()
    session = requests.Session()
    if args.proxy_auth_file:
        auth = json.loads(Path(args.proxy_auth_file).read_text())
        proxy = "http://" + quote(auth["username"], safe="") + ":" + quote(auth["password"], safe="") + "@127.0.0.1:18089"
        session.trust_env = False
        session.proxies.update(http=proxy, https=proxy)
    response = session.get("https://huggingface.co/api/datasets/bop-benchmark/ycbv?blobs=true", timeout=60)
    response.raise_for_status()
    metadata = response.json()
    entry = next(r for r in metadata["siblings"] if r["rfilename"] == args.archive)
    url = f"https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{metadata['sha']}/{args.archive}"
    reader = RangeFile(session, url, entry["lfs"]["size"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = {f"{s:06d}" for s in args.scenes}
    folders = {"rgb", "depth"} | ({"mask", "mask_visib"} if args.include_masks else set())
    files = []
    with zipfile.ZipFile(reader) as archive:
        selected = []
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 3 or parts[0] != "test" or parts[1] not in names or info.is_dir():
                continue
            if parts[2] in folders or (len(parts) == 3 and parts[2].endswith(".json")):
                selected.append(info)
        if not selected:
            raise ValueError("requested scenes not found in official archive")
        print(json.dumps({"selected_files": len(selected), "archive_bytes": reader.size,
                          "scenes": args.scenes, "revision": metadata["sha"]}), flush=True)
        for index, info in enumerate(sorted(selected, key=lambda item: item.header_offset)):
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe archive path")
            target = output.joinpath(*relative.parts)
            if target.exists():
                content = target.read_bytes()
                if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                    raise ValueError("existing output differs from the pinned archive")
            else:
                content = archive.read(info)  # ZIP verifies the full member CRC.
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix+".partial")
                temporary.write_bytes(content)
                temporary.replace(target)
            files.append({"path": str(relative), "bytes": len(content), "zip_crc32": info.CRC,
                          "sha256": hashlib.sha256(content).hexdigest()})
            if index % 100 == 0 or index + 1 == len(selected):
                print(json.dumps({"files_done": index+1, "total": len(selected),
                                  "downloaded_bytes": reader.downloaded}), flush=True)
    report = {
        "source_repository": "bop-benchmark/ycbv", "revision": metadata["sha"],
        "archive": args.archive, "archive_publisher_sha256": entry["lfs"]["sha256"],
        "whole_archive_sha256_verified": False,
        "verification": "pinned official ZIP; each extracted member CRC checked; extracted SHA256 recorded",
        "downloaded_bytes": reader.downloaded, "scenes": args.scenes, "files": files,
    }
    manifest = output/f"download-{Path(args.archive).stem}-{'-'.join(map(str,args.scenes))}.json"
    manifest.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"complete": True, "files": len(files), "manifest": str(manifest)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as error:
        # Network exceptions can contain authenticated proxy URLs.
        print(type(error).__name__, "network request failed; credentials omitted", flush=True)
        raise SystemExit(1)
