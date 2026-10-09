"""Local-only import/export for CyberArena SQLite memory. Keep packages private."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from utils.experience_memory import ExperienceMemory


def main() -> int:
    parser=argparse.ArgumentParser(description="Export/import a CyberArena memory JSON package")
    parser.add_argument("--db",required=True,type=Path,help="Existing SQLite database file")
    commands=parser.add_subparsers(dest="command",required=True)
    export_parser=commands.add_parser("export",help="Export experiences and replay transitions")
    export_parser.add_argument("destination",type=Path,help="New JSON package path")
    export_parser.add_argument("--overwrite",action="store_true",help="Explicitly replace an export")
    import_parser=commands.add_parser("import",help="Merge a compatible package transactionally")
    import_parser.add_argument("source",type=Path,help="Versioned memory JSON package")
    args=parser.parse_args()
    database=args.db.expanduser().resolve()
    if not database.is_file(): parser.error(f"database must already exist: {database}")
    package_path=(args.destination if args.command=="export" else args.source).expanduser().resolve()
    if package_path==database: parser.error("package path must not be the active SQLite database")
    memory=None
    try:
        memory=ExperienceMemory(database)
        result=(memory.export_json(package_path,overwrite=args.overwrite)
                if args.command=="export" else memory.import_json(package_path))
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print(f"Memory transfer failed safely: {exc}",file=sys.stderr)
        return 2
    finally:
        if memory is not None: memory.close()


if __name__=="__main__":
    raise SystemExit(main())
