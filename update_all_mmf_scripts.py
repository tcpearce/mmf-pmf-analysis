#!/usr/bin/env python3
"""
Batch update script to modify all MMF analysis scripts to use corrected data paths.
"""

import re
from pathlib import Path

# Scripts that need updating and their patterns to replace
SCRIPTS_TO_UPDATE = {
    'check_units.py': [
        (r"parquet_file = Path\('mmf_parquet'\)", "parquet_file = Path('mmf_parquet_final')"),
        (r"Path\(f'mmf_data/\{station\}/raw/'\)", "Path(f'mmf_data_corrected/{station}/raw/')")
    ],
    'cleanup_mmf9_data.py': [
        (r"parquet_file = Path\('mmf_parquet/MMF9_combined_data.parquet'\)", 
         "parquet_file = Path('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')"),
        (r"backup_dir = Path\('mmf_parquet/backup'\)", 
         "backup_dir = Path('mmf_parquet_final/backup')")
    ],
    'examine_units_safely.py': [
        (r'raw_dir = Path\(f"mmf_data/\{station\}/raw"\)', 
         'raw_dir = Path(f"mmf_data_corrected/{station}/raw")'),
        (r'parquet_file = Path\("mmf_parquet/MMF1_combined_data.parquet"\)',
         'parquet_file = Path("mmf_parquet_final/MMF1_Cemetery_Road_combined_data.parquet")')
    ],
    'download_mmf_data.py': [
        (r"base_dir = Path\('mmf_data'\)", "base_dir = Path('mmf_data_corrected')")
    ],
    'organize_mmf_data.py': [
        (r'def __init__\(self, base_dir="mmf_data"\)', 'def __init__(self, base_dir="mmf_data_corrected")')
    ],
    'validate_extended_data.py': [
        (r"'mmf_parquet'", "'mmf_parquet_final'")
    ],
    'check_excel_units.py': [
        (r"raw_dir = Path\(f'mmf_data/\{station\}/raw'\)", 
         "raw_dir = Path(f'mmf_data_corrected/{station}/raw')")
    ]
}

# Scripts that need import statements added
SCRIPTS_NEEDING_IMPORTS = [
    'pmf_source_apportionment_fixed.py',
    'pmf_source_apportionment_weekly.py', 
    'pmf_complaint_integration.py'
]

def add_mmf_config_import(script_path):
    """Add mmf_config import to a script."""
    print(f"Adding mmf_config import to {script_path.name}...")
    
    try:
        content = script_path.read_text(encoding='utf-8')
        
        # Check if already imported
        if 'from mmf_config import' in content:
            print(f"  ✅ {script_path.name} already has mmf_config import")
            return True
            
        # Find the last import statement
        lines = content.split('\n')
        import_line_idx = -1
        
        for i, line in enumerate(lines):
            if (line.strip().startswith('import ') or 
                line.strip().startswith('from ') or
                line.strip().startswith('#') and 'import' not in line.lower()):
                import_line_idx = i
                
        if import_line_idx == -1:
            print(f"  ⚠️  Could not find import section in {script_path.name}")
            return False
            
        # Insert the import
        insert_idx = import_line_idx + 1
        lines.insert(insert_idx, "from mmf_config import get_mmf_parquet_file, get_corrected_mmf_files")
        
        # Write back
        script_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"  ✅ Added mmf_config import to {script_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error updating {script_path.name}: {e}")
        return False

def update_script_patterns(script_path, patterns):
    """Update a script with pattern replacements."""
    print(f"Updating {script_path.name} with {len(patterns)} patterns...")
    
    try:
        content = script_path.read_text(encoding='utf-8')
        original_content = content
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
            
        if content != original_content:
            script_path.write_text(content, encoding='utf-8')
            print(f"  ✅ Updated {script_path.name}")
            return True
        else:
            print(f"  ℹ️  No changes needed for {script_path.name}")
            return True
            
    except Exception as e:
        print(f"  ❌ Error updating {script_path.name}: {e}")
        return False

def main():
    print("MMF SCRIPTS BATCH UPDATE")
    print("=" * 40)
    print("Updating all scripts to use corrected MMF data paths...")
    print()
    
    current_dir = Path('.')
    
    # Update scripts with pattern replacements
    success_count = 0
    total_count = 0
    
    for script_name, patterns in SCRIPTS_TO_UPDATE.items():
        script_path = current_dir / script_name
        total_count += 1
        
        if not script_path.exists():
            print(f"⚠️  Script not found: {script_name}")
            continue
            
        if update_script_patterns(script_path, patterns):
            success_count += 1
    
    print()
    
    # Add imports to PMF scripts
    for script_name in SCRIPTS_NEEDING_IMPORTS:
        script_path = current_dir / script_name
        total_count += 1
        
        if not script_path.exists():
            print(f"⚠️  Script not found: {script_name}")
            continue
            
        if add_mmf_config_import(script_path):
            success_count += 1
    
    print()
    print("BATCH UPDATE SUMMARY:")
    print("-" * 25)
    print(f"Scripts processed: {total_count}")
    print(f"Successfully updated: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("✅ All scripts updated successfully!")
    else:
        print("⚠️  Some scripts had issues - review output above")
    
    print()
    print("IMPORTANT REMINDERS:")
    print("- Updated scripts now use mmf_parquet_final/ and mmf_data_corrected/")
    print("- MMF numbers now point to correct stations:")
    print("  • MMF1 → Cemetery Road")
    print("  • MMF2 → Silverdale Pumping Station") 
    print("  • MMF6 → Fire Station")
    print("  • MMF9 → Galingale View")
    print("  • Maries_Way → No MMF number")

if __name__ == "__main__":
    main()