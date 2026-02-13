#!/usr/bin/env python3
"""
Unit tests for subtitle-enhanced profanity detection.
Tests the core logic without requiring actual video/audio files.
"""

import unittest
import tempfile
import os
import sys

# Mock heavy dependencies that aren't needed for unit tests
from unittest.mock import MagicMock
sys.modules['faster_whisper'] = MagicMock()
sys.modules['ctranslate2'] = MagicMock()

from teodor.defuse import (
    parse_srt_for_profanity,
    merge_profanity_results,
    check_subtitles_for_profanity,
    PROFANITY_PATTERNS
)
import re


class TestProfanityPatterns(unittest.TestCase):
    """Test that profanity patterns match expected words."""

    def test_fuck_variations(self):
        pattern = re.compile(PROFANITY_PATTERNS[0], re.IGNORECASE)
        # Should match - includes compound words
        self.assertIsNotNone(pattern.search("fuck"))
        self.assertIsNotNone(pattern.search("Fuck"))
        self.assertIsNotNone(pattern.search("FUCK"))
        self.assertIsNotNone(pattern.search("fucking"))
        self.assertIsNotNone(pattern.search("fucker"))
        self.assertIsNotNone(pattern.search("fucked"))
        self.assertIsNotNone(pattern.search("motherfucker"))
        # Should not match
        self.assertIsNone(pattern.search("duck"))
        self.assertIsNone(pattern.search("luck"))

    # test_shit_variations skipped - shit pattern is currently commented out in PROFANITY_PATTERNS


class TestParseSrtForProfanity(unittest.TestCase):
    """Test SRT parsing functionality."""

    def setUp(self):
        """Create a temporary SRT file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.srt_file = os.path.join(self.temp_dir, "test.srt")

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.srt_file):
            os.remove(self.srt_file)
        os.rmdir(self.temp_dir)

    def test_parse_srt_with_profanity(self):
        """Test parsing SRT with profanity."""
        srt_content = """1
00:00:10,000 --> 00:00:12,000
What the fuck is that?

2
00:00:15,000 --> 00:00:17,000
I don't know, man.

3
00:00:20,000 --> 00:00:23,000
This is fucking crazy!
"""
        with open(self.srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        results = parse_srt_for_profanity(self.srt_file)

        self.assertEqual(len(results), 2)
        # First profanity: "What the fuck is that?" (22 chars, 10.0-12.0s)
        # "fuck" at chars 9-13 → interpolated then ±0.3s buffer
        self.assertEqual(results[0]['word'], 'fuck')
        self.assertAlmostEqual(results[0]['start'], 10.52, delta=0.1)
        self.assertAlmostEqual(results[0]['end'], 11.48, delta=0.1)
        # Second profanity: "This is fucking crazy!" (22 chars, 20.0-23.0s)
        # "fucking" at chars 8-15 → interpolated then ±0.3s buffer
        self.assertEqual(results[1]['word'], 'fucking')
        self.assertAlmostEqual(results[1]['start'], 20.79, delta=0.1)
        self.assertAlmostEqual(results[1]['end'], 22.35, delta=0.1)

    def test_parse_srt_no_profanity(self):
        """Test parsing SRT without profanity."""
        srt_content = """1
00:00:10,000 --> 00:00:12,000
Hello, how are you?

2
00:00:15,000 --> 00:00:17,000
I'm doing well, thanks.
"""
        with open(self.srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        results = parse_srt_for_profanity(self.srt_file)
        self.assertEqual(len(results), 0)

    def test_parse_srt_multiple_profanity_same_line(self):
        """Test parsing SRT with multiple profanity in same subtitle.
        Note: only 'fuck' pattern is currently active (shit is commented out).
        """
        srt_content = """1
00:00:10,000 --> 00:00:15,000
Fuck this shit, I'm done!
"""
        with open(self.srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        results = parse_srt_for_profanity(self.srt_file)
        # Only fuck pattern is active, shit is commented out
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'Fuck')


    def test_interpolation_gives_distinct_times(self):
        """Test that multiple profanity words in same line get distinct timestamps."""
        srt_content = """1
00:00:10,000 --> 00:00:16,000
What the fuck? Are you fucking kidding me?
"""
        with open(self.srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        results = parse_srt_for_profanity(self.srt_file)
        self.assertEqual(len(results), 2)
        # The two words should have different start/end times
        self.assertNotAlmostEqual(results[0]['start'], results[1]['start'], places=1)
        self.assertNotAlmostEqual(results[0]['end'], results[1]['end'], places=1)
        # First word ("fuck") should be earlier than second ("fucking")
        self.assertLess(results[0]['start'], results[1]['start'])


class TestMergeProfanityResults(unittest.TestCase):
    """Test merging of Whisper and subtitle results."""

    def test_merge_with_no_overlap(self):
        """Test merging when Whisper missed all subtitle profanity."""
        whisper_swears = []  # Whisper found nothing
        subtitle_swears = [
            {'word': 'fuck', 'start': 10.0, 'end': 12.0, 'subtitle_text': 'test', 'source': 'subtitle'},
            {'word': 'shit', 'start': 20.0, 'end': 22.0, 'subtitle_text': 'test', 'source': 'subtitle'},
        ]

        results = merge_profanity_results(whisper_swears, subtitle_swears)

        self.assertEqual(len(results), 2)
        # Check that subtitle fallbacks were added with buffers
        self.assertEqual(results[0][0], 'fuck')
        self.assertAlmostEqual(results[0][1], 9.8, places=1)  # start - 0.2
        self.assertAlmostEqual(results[0][2], 12.3, places=1)  # end + 0.3

    def test_merge_with_full_overlap(self):
        """Test merging when Whisper found all subtitle profanity."""
        whisper_swears = [
            ('fuck', 10.5, 11.0),  # Within subtitle window 10-12
            ('shit', 20.3, 20.8),  # Within subtitle window 20-22
        ]
        subtitle_swears = [
            {'word': 'fuck', 'start': 10.0, 'end': 12.0, 'subtitle_text': 'test', 'source': 'subtitle'},
            {'word': 'shit', 'start': 20.0, 'end': 22.0, 'subtitle_text': 'test', 'source': 'subtitle'},
        ]

        results = merge_profanity_results(whisper_swears, subtitle_swears)

        # Should only have Whisper results (no duplicates from subtitles)
        self.assertEqual(len(results), 2)
        # First result should be Whisper's timing
        self.assertEqual(results[0][0], 'fuck')
        self.assertAlmostEqual(results[0][1], 10.5, places=1)

    def test_merge_with_partial_overlap(self):
        """Test merging when Whisper found some but not all."""
        whisper_swears = [
            ('fuck', 10.5, 11.0),  # Found this one
        ]
        subtitle_swears = [
            {'word': 'fuck', 'start': 10.0, 'end': 12.0, 'subtitle_text': 'test', 'source': 'subtitle'},
            {'word': 'shit', 'start': 50.0, 'end': 52.0, 'subtitle_text': 'test', 'source': 'subtitle'},
        ]

        results = merge_profanity_results(whisper_swears, subtitle_swears)

        self.assertEqual(len(results), 2)
        # First is Whisper's
        self.assertEqual(results[0][0], 'fuck')
        self.assertAlmostEqual(results[0][1], 10.5, places=1)
        # Second is subtitle fallback
        self.assertEqual(results[1][0], 'shit')
        self.assertAlmostEqual(results[1][1], 49.8, places=1)  # 50.0 - 0.2

    def test_merge_sorted_by_time(self):
        """Test that merged results are sorted by start time."""
        whisper_swears = [
            ('fuck', 30.0, 30.5),
        ]
        subtitle_swears = [
            {'word': 'shit', 'start': 10.0, 'end': 12.0, 'subtitle_text': 'test', 'source': 'subtitle'},
        ]

        results = merge_profanity_results(whisper_swears, subtitle_swears)

        self.assertEqual(len(results), 2)
        # Should be sorted by start time
        self.assertTrue(results[0][1] < results[1][1])
        self.assertEqual(results[0][0], 'shit')  # Earlier one first
        self.assertEqual(results[1][0], 'fuck')  # Later one second


    def test_merge_deduplicates_overlapping(self):
        """Test that overlapping entries are merged into one."""
        whisper_swears = [
            ('fuck', 10.5, 11.0),
        ]
        subtitle_swears = [
            # This subtitle entry overlaps with the whisper detection
            {'word': 'fuck', 'start': 10.4, 'end': 11.1, 'subtitle_text': 'test', 'source': 'subtitle'},
            # This one is far away, should not be merged
            {'word': 'fuck', 'start': 50.0, 'end': 51.0, 'subtitle_text': 'test', 'source': 'subtitle'},
        ]

        results = merge_profanity_results(whisper_swears, subtitle_swears, tolerance=0.5)

        # The close entries should be merged, leaving 2 total (not 3)
        # The second subtitle entry at 50s should remain separate
        self.assertEqual(len(results), 2)
        # First should be whisper-sourced (preferred over subtitle)
        self.assertAlmostEqual(results[0][1], 10.5, places=1)
        # Second is the far-away subtitle fallback
        self.assertAlmostEqual(results[1][1], 49.8, places=1)

    def test_merge_deduplicates_adjacent_subtitle_entries(self):
        """Test that adjacent subtitle-only entries get merged."""
        whisper_swears = []
        subtitle_swears = [
            {'word': 'fuck', 'start': 10.0, 'end': 10.8, 'subtitle_text': 'test', 'source': 'subtitle'},
            {'word': 'fucking', 'start': 10.5, 'end': 11.2, 'subtitle_text': 'test', 'source': 'subtitle'},
        ]

        results = merge_profanity_results(whisper_swears, subtitle_swears)

        # Should be merged into one entry since they overlap
        self.assertEqual(len(results), 1)


class TestCheckSubtitlesForProfanity(unittest.TestCase):
    """Test quick subtitle profanity check."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_check_external_srt_with_profanity(self):
        """Test checking external SRT with profanity."""
        video_file = os.path.join(self.temp_dir, "movie.mkv")
        srt_file = os.path.join(self.temp_dir, "movie.srt")

        # Create dummy video file (just for path)
        with open(video_file, 'w') as f:
            f.write("dummy")

        # Create SRT with profanity
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write("1\n00:00:10,000 --> 00:00:12,000\nWhat the fuck?\n")

        has_profanity, result_path = check_subtitles_for_profanity(
            video_file,
            subtitles_exist=False,
            external_srt_exists=True
        )

        self.assertTrue(has_profanity)
        self.assertEqual(result_path, srt_file)

    def test_check_external_srt_without_profanity(self):
        """Test checking external SRT without profanity."""
        video_file = os.path.join(self.temp_dir, "movie.mkv")
        srt_file = os.path.join(self.temp_dir, "movie.srt")

        # Create dummy video file
        with open(video_file, 'w') as f:
            f.write("dummy")

        # Create clean SRT
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write("1\n00:00:10,000 --> 00:00:12,000\nHello there!\n")

        has_profanity, result_path = check_subtitles_for_profanity(
            video_file,
            subtitles_exist=False,
            external_srt_exists=True
        )

        self.assertFalse(has_profanity)
        self.assertEqual(result_path, srt_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)
