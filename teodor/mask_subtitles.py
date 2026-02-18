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

# Base profanity patterns (always active)
PROFANITY_PATTERNS = [
    r'f+u+c+k+',
]


def word_to_pattern(word):
    """Convert a plain word to a flexible profanity regex pattern.

    Deduplicates consecutive repeated letters so that e.g. "goddamn" and
    "godamn" produce the same pattern.  Each unique letter becomes ``<letter>+``
    so one or more repetitions are matched.
    """
    chars = []
    for c in word.lower():
        if not chars or c != chars[-1]:
            chars.append(c)
    return ''.join(f'{c}+' for c in chars)


def _build_pattern(patterns):
    """Build a compiled regex from a list of pattern strings."""
    combined = '|'.join(f'({p})' for p in patterns)
    return re.compile(combined, re.IGNORECASE)


# Default compiled pattern (f-word only)
PROFANITY_PATTERN = _build_pattern(PROFANITY_PATTERNS)


def mask_profanity(text: str, pattern=None) -> str:
    """Replace profanity matches with asterisks of the same length."""
    if pattern is None:
        pattern = PROFANITY_PATTERN
    def replace_match(match):
        return '*' * len(match.group(0))
    return pattern.sub(replace_match, text)


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


def process_subtitle_file(input_path: Path, output_path: Path, pattern=None) -> int:
    """Process a subtitle file and mask profanity. Returns count of replacements."""
    if pattern is None:
        pattern = PROFANITY_PATTERN
    # Read the file
    content, encoding = read_subtitle_file(input_path)

    # Count matches before replacing
    matches = pattern.findall(content)
    count = len(matches)

    if count == 0:
        print(f"No profanity found in {input_path.name}")
        return 0

    # Mask profanity
    masked_content = mask_profanity(content, pattern=pattern)

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
    parser.add_argument(
        '--swears', nargs='*', default=[],
        help='Additional profanity words to mask (e.g. --swears shit damn)'
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

    # Build extended pattern if user supplied extra words
    if args.swears:
        all_patterns = PROFANITY_PATTERNS + [word_to_pattern(w) for w in args.swears]
        pattern = _build_pattern(all_patterns)
    else:
        pattern = PROFANITY_PATTERN

    try:
        count = process_subtitle_file(input_path, output_path, pattern=pattern)
        if count > 0:
            print(f"Output written to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
