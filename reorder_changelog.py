#!/usr/bin/env python3
"""
CHANGELOG Date-Time Reordering Script

This script detects date-time stamps in the CHANGELOG file (format: YYYY-MM-DD HH:MM)
and reorders all entries by date-time with newest entries first.

Preserves all original content while ensuring chronological order.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


def parse_changelog_entries(changelog_content: str) -> List[Tuple[Optional[datetime], str]]:
    """
    Parse changelog content and extract entries with their timestamps.
    
    Args:
        changelog_content: Full content of the changelog file
        
    Returns:
        List of tuples (datetime_obj, entry_content) ordered by appearance
    """
    # Pattern to match date-time stamps like "2025-01-26 16:00"
    datetime_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})'
    
    # Split content by date-time patterns while keeping the delimiters
    parts = re.split(f'(?=## {datetime_pattern})', changelog_content)
    
    entries = []
    header_content = ""
    
    for i, part in enumerate(parts):
        if not part.strip():
            continue
            
        # Check if this part starts with a date-time stamp
        datetime_match = re.search(f'^## {datetime_pattern}', part, re.MULTILINE)
        
        if datetime_match:
            # Extract the date-time string
            datetime_str = datetime_match.group(1)
            
            try:
                # Parse the datetime (handle both HH:MM and H:MM formats)
                dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
            except ValueError:
                # If parsing fails, try without leading zero in hour
                try:
                    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
                except ValueError:
                    print(f"Warning: Could not parse datetime '{datetime_str}', treating as undated")
                    dt = None
            
            entries.append((dt, part))
        else:
            # This is either the header or undated content
            if i == 0:
                header_content = part
            else:
                # Undated content - add with None timestamp
                entries.append((None, part))
    
    return header_content, entries


def reorder_entries(header: str, entries: List[Tuple[Optional[datetime], str]]) -> str:
    """
    Reorder entries with newest first, preserving undated entries at appropriate positions.
    
    Args:
        header: Header content before any dated entries
        entries: List of (datetime, content) tuples
        
    Returns:
        Reordered changelog content
    """
    # Separate dated and undated entries
    dated_entries = [(dt, content) for dt, content in entries if dt is not None]
    undated_entries = [content for dt, content in entries if dt is None]
    
    # Sort dated entries by datetime (newest first)
    dated_entries.sort(key=lambda x: x[0], reverse=True)
    
    # Reconstruct the changelog
    result_parts = []
    
    # Add header if it exists
    if header.strip():
        result_parts.append(header.rstrip())
        result_parts.append("")  # Add spacing after header
    
    # Add dated entries (newest first)
    for i, (dt, content) in enumerate(dated_entries):
        if i > 0:
            result_parts.append("")  # Add spacing between entries
        result_parts.append(content.rstrip())
    
    # Add undated entries at the end
    for content in undated_entries:
        result_parts.append("")
        result_parts.append(content.rstrip())
    
    return "\n".join(result_parts)


def create_backup(file_path: Path) -> Path:
    """
    Create a backup of the original file.
    
    Args:
        file_path: Path to the original file
        
    Returns:
        Path to the backup file
    """
    backup_path = file_path.with_suffix(f'{file_path.suffix}.backup')
    backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
    return backup_path


def validate_content_preservation(original: str, reordered: str) -> bool:
    """
    Validate that no content was lost during reordering.
    
    Args:
        original: Original changelog content
        reordered: Reordered changelog content
        
    Returns:
        True if all content is preserved
    """
    # Remove extra whitespace and normalize line endings for comparison
    orig_normalized = re.sub(r'\s+', ' ', original.strip())
    reord_normalized = re.sub(r'\s+', ' ', reordered.strip())
    
    # Check if all non-whitespace content is preserved
    orig_words = set(orig_normalized.split())
    reord_words = set(reord_normalized.split())
    
    missing_words = orig_words - reord_words
    extra_words = reord_words - orig_words
    
    if missing_words:
        print(f"Warning: Missing words detected: {list(missing_words)[:10]}...")
        return False
    
    if extra_words:
        print(f"Warning: Extra words detected: {list(extra_words)[:10]}...")
        return False
    
    return True


def main():
    """Main execution function."""
    changelog_path = Path("CHANGELOG.md")
    
    if not changelog_path.exists():
        print(f"Error: {changelog_path} not found in current directory")
        sys.exit(1)
    
    print(f"Processing {changelog_path}...")
    
    try:
        # Read the original changelog
        original_content = changelog_path.read_text(encoding='utf-8')
        print(f"Original file size: {len(original_content):,} characters")
        
        # Create backup
        backup_path = create_backup(changelog_path)
        print(f"Backup created: {backup_path}")
        
        # Parse entries
        header, entries = parse_changelog_entries(original_content)
        
        dated_entries = [e for e in entries if e[0] is not None]
        undated_entries = [e for e in entries if e[0] is None]
        
        print(f"Found {len(dated_entries)} dated entries and {len(undated_entries)} undated entries")
        
        if dated_entries:
            oldest_date = min(dt for dt, _ in dated_entries)
            newest_date = max(dt for dt, _ in dated_entries)
            print(f"Date range: {oldest_date.strftime('%Y-%m-%d %H:%M')} to {newest_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Reorder entries
        reordered_content = reorder_entries(header, entries)
        print(f"Reordered file size: {len(reordered_content):,} characters")
        
        # Validate content preservation
        if not validate_content_preservation(original_content, reordered_content):
            print("ERROR: Content validation failed! Check backup file.")
            sys.exit(1)
        
        print("✅ Content validation passed - no data loss detected")
        
        # Write the reordered changelog
        changelog_path.write_text(reordered_content, encoding='utf-8')
        print(f"✅ Successfully reordered {changelog_path} with newest entries first")
        
        # Display the first few entries for verification
        lines = reordered_content.split('\n')
        print("\nFirst few lines of reordered changelog:")
        for i, line in enumerate(lines[:20]):
            if line.strip():
                print(f"  {line}")
            if i > 15 and any(re.match(r'## \d{4}-\d{2}-\d{2}', l) for l in lines[i:i+5]):
                print("  ...")
                break
        
    except Exception as e:
        print(f"Error processing changelog: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()