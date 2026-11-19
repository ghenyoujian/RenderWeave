"""Print actual C ABI/library information without importing a tensor framework."""
import argparse
import json
import platform
import sys

from . import Runtime, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="explicit installed librw_core path")
    args = parser.parse_args()
    with Runtime(library=args.library) as runtime:
        print(json.dumps({
            "project_id": "renderweave", "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "python_package_version": __version__, "library_path": str(runtime.library_path),
            "capabilities": runtime.capabilities(),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
