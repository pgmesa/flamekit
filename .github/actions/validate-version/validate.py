
import re
import ast
import sys
import argparse
from pathlib import Path


SEMVER_RE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)"
    r"(?:-([0-9A-Za-z.-]+))?"
    r"(?:\+([0-9A-Za-z.-]+))?$"
)


def extract_version(version_file: Path) -> str:
    tree = ast.parse(version_file.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "__version__":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"__version__ not found in {version_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--version-file", required=True)
    args = parser.parse_args()

    tag = args.tag
    version_file = Path(args.version_file)

    if not tag.startswith("v"):
        print("ERROR: tag must start with 'v' (e.g. v1.2.3)")
        sys.exit(1)

    tag_version = tag[1:]

    if not SEMVER_RE.match(tag_version):
        print(f"ERROR: tag version is not valid semver: {tag_version}")
        sys.exit(1)

    if not version_file.exists():
        print(f"ERROR: version file not found: {version_file}")
        sys.exit(1)

    try:
        code_version = extract_version(version_file)
    except Exception as e:
        print(f"ERROR: failed to extract __version__: {e}")
        sys.exit(1)

    if code_version != tag_version:
        print(f"ERROR: version mismatch: tag={tag_version}, code={code_version}")
        sys.exit(1)

    print(f"OK: tag and code version match: {tag_version}")


if __name__ == "__main__":
    main()