"""Explicitly download/export EvalPlus data for preparation; never execute solutions."""
import argparse
import json
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Private source export outside prepared public directories')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    from evalplus.data import get_human_eval_plus
    records = get_human_eval_plus()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import os
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as file:
        for record in records.values():
            file.write(json.dumps(record)+'\n')
    print('Exported', len(records), 'private source records to', args.output)
