#!/usr/bin/env python3
"""
mask-subtitles.py - Mask profanity in subtitle files

Replaces instances of "fuck" (and variations) with asterisks while preserving
the subtitle file structure and encoding.

Usage:
    python3 mask-subtitles.py -i subtitle.srt
    python3 mask-subtitles.py -i subtitle.srt -o subtitle-clean.srt
"""

import argparse
import re
import shutil
from pathlib import Path

# Pattern to match fuck and variations (fucking, fucker, motherfucker, etc.)
PROFANITY_PATTERN = re.compile(r'(f+u+c+k+)', re.IGNORECASE)


def mask_profanity(text: str) -> str:
    """Replace profanity matches with asterisks of the same length."""
    def replace_match(match):
        return '*' * len(match.group(0))
    return PROFANITY_PATTERN.sub(replace_match, text)


def read_subtitle_file(filepath: Path) -> tuple[str, str]:
    """Read subtitle file, trying different encodings. Returns (content, encoding)."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            return content, encoding
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {filepath} with any supported encoding")


def process_subtitle_file(input_path: Path, output_path: Path) -> int:
    """Process a subtitle file and mask profanity. Returns count of replacements."""
    # Read the file
    content, encoding = read_subtitle_file(input_path)

    # Count matches before replacing
    matches = PROFANITY_PATTERN.findall(content)
    count = len(matches)

    if count == 0:
        print(f"No profanity found in {input_path.name}")
        return 0

    # Mask profanity
    masked_content = mask_profanity(content)

    # Write to output file (use same encoding as input)
    with open(output_path, 'w', encoding=encoding) as f:
        f.write(masked_content)

    print(f"Masked {count} instance(s) of profanity in {input_path.name}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description='Mask profanity in subtitle files'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input subtitle file (.srt)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file (default: input-CLEAN.srt)'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify the input file in place (creates backup first)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    if args.in_place:
        # Create backup
        backup_path = input_path.with_suffix(input_path.suffix + '.bak')
        shutil.copy2(input_path, backup_path)
        print(f"Backup created: {backup_path}")
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        # Default: input-CLEAN.srt
        output_path = input_path.with_stem(input_path.stem + '-CLEAN')

    try:
        count = process_subtitle_file(input_path, output_path)
        if count > 0:
            print(f"Output written to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
