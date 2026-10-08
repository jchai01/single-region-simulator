"""
Are the files in this folder the latest delivery?

The pipeline is ~20 files that must move together; copying only some of them gives
confusing errors (a TypeError about an unexpected keyword, a banner from an old
version). This compares every file against MANIFEST.json (the SHA-256 of each file in the
latest delivery) and says exactly which are up to date, which differ, and which are missing.

Uses only the standard library, so it works even when the project's own modules are
mismatched.

    python check_install.py            # check the folder this script is in
    python check_install.py --dir PATH

"differs" means the content is not byte-identical to the delivery: either the file is older,
or you edited it on purpose (for example nsga2_scenario_search.py if you changed the
population settings). If you did not edit it, copy the delivered one over it.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    manifest_path = os.path.join(args.dir, "MANIFEST.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"MANIFEST.json not found in {args.dir}: copy it from the latest delivery next to this script.")
    manifest = json.load(open(manifest_path))
    ok, differs, missing = [], [], []
    for name, info in sorted(manifest["files"].items()):
        path = os.path.join(args.dir, name)
        if not os.path.exists(path):
            missing.append(name)
        elif sha256(path) == info["sha256"]:
            ok.append(name)
        else:
            differs.append(name)

    print(f"Delivery: {manifest.get('delivery', '(unnamed)')}  ({len(manifest['files'])} files)")
    print(f"  up to date: {len(ok)}    differ: {len(differs)}    missing: {len(missing)}\n")
    for name in differs:
        print(f"  DIFFERS  {name}   -- {manifest['files'][name].get('role', '')}")
    for name in missing:
        print(f"  MISSING  {name}   -- {manifest['files'][name].get('role', '')}")
    if differs or missing:
        print("\nCopy the DIFFERS/MISSING files from the latest delivery into this folder (skip any you edited on purpose), "
              "then rerun this check.")
        sys.exit(1)
    print("All files match the latest delivery.")


if __name__ == "__main__":
    main()
