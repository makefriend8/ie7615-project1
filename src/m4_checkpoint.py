"""Restore or verify the original classifier required by the frozen M4 policy."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def restore(url=None,source=None,output=None):
    policy=json.loads((ROOT/"m4_artifacts/integration_policy.json").read_text())
    expected=policy["checkpoint_sha256"]["classifier"]
    target=Path(output or ROOT/"runs/m4/classifier13/best.weights.h5").resolve()
    if target.exists():
        if digest(target)!=expected:raise ValueError(f"Preserving mismatched checkpoint: {target}")
        print(f"Verified original classifier: {target}")
        return target
    if not url and not source:raise FileNotFoundError("Original classifier is missing; use --source for an existing checkpoint copy or --url for an HTTPS checkpoint file")
    if url and urlparse(url).scheme!="https":raise ValueError("Use an HTTPS checkpoint URL")
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_name(target.name+".download")
    if partial.exists():raise FileExistsError(f"Preserving incomplete download: {partial}")
    # Keep the temporary download on failure for diagnosis. Never change the policy.
    with partial.open("xb") as destination:
        if source:
            with Path(source).open("rb") as origin:shutil.copyfileobj(origin,destination)
        else:
            with urlopen(url,timeout=60) as origin:shutil.copyfileobj(origin,destination)
    if digest(partial)!=expected:raise ValueError(f"Restored file differs from frozen checkpoint; preserved at {partial}")
    if target.exists():raise FileExistsError(f"Preserving concurrently created target: {target}")
    partial.rename(target)
    print(f"Restored and verified original classifier: {target}")
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--source",type=Path,help="Path to an existing copy of the original best.weights.h5")
    group.add_argument("--url",help="Optional HTTPS URL of the original checkpoint file")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    restore(args.url,args.source,args.output)


if __name__=="__main__":main()
