import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Args:
        path: Input JSONL file path.

    Returns:
        Parsed JSON objects.
    """
    records = []
    with path.open() as file:
        for line in file:
            records.append(json.loads(line))
    return records


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as a formatted JSON array.

    Args:
        path: Output JSON file path.
        records: Records to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file to a formatted JSON array.",
    )
    parser.add_argument("input", type=Path, help="JSONL input path")
    parser.add_argument("output", type=Path, help="Formatted JSON output path")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    write_json(args.output, records)
    print(f"exported {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
