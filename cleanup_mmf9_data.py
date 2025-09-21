#!/usr/bin/env python3
"""
Clean up MMF9 parquet file by removing empty Unnamed columns.
These columns were likely created from empty columns in the original Excel file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import shutil
from datetime import datetime

def cleanup_mmf9_parquet():
    """Remove unnamed/empty columns from MMF9 parquet file."""
    
    parquet_file = Path('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
    backup_dir = Path('mmf_parquet_final/backup')
    
    if not parquet_file.exists():
        print(f"❌ File not found: {parquet_file}")
        return
    
    print("🔧 Cleaning up MMF9 parquet file...")
    print(f"📁 Processing: {parquet_file}")
    
    # Create backup
    backup_dir.mkdir(exist_ok=True)
    backup_file = backup_dir / f"MMF9_combined_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    shutil.copy2(parquet_file, backup_file)
    print(f"💾 Backup created: {backup_file}")
    
    # Load data
    df = pd.read_parquet(parquet_file)
    print(f"📊 Original data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Identify columns to remove
    columns_to_remove = []
    columns_to_keep = []
    
    for col in df.columns:
        if col.startswith('Unnamed:'):
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 95:  # Remove columns with >95% missing data
                columns_to_remove.append(col)
                print(f"  ❌ Removing: {col} ({missing_pct:.1f}% missing)")
            else:
                columns_to_keep.append(col)
                print(f"  ⚠️  Keeping: {col} ({missing_pct:.1f}% missing) - has some data")
        else:
            columns_to_keep.append(col)
    
    if not columns_to_remove:
        print("✅ No empty unnamed columns found. File is already clean.")
        return
    
    print(f"\n📋 Removing {len(columns_to_remove)} empty columns")
    print(f"📋 Keeping {len(columns_to_keep)} columns with data")
    
    # Create cleaned dataframe
    df_clean = df[columns_to_keep].copy()
    
    print(f"📊 Cleaned data: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
    
    # Read original metadata to preserve it
    pq_file = pq.ParquetFile(parquet_file)
    original_metadata = pq_file.metadata.metadata
    
    # Create new metadata preserving the original
    metadata_dict = {}
    if original_metadata:
        for key, value in original_metadata.items():
            key_str = key.decode() if isinstance(key, bytes) else str(key)
            value_str = value.decode() if isinstance(value, bytes) else str(value)
            metadata_dict[key_str] = value_str
    
    # Add cleanup information
    metadata_dict['cleanup_date'] = datetime.now().isoformat()
    metadata_dict['columns_removed'] = str(len(columns_to_remove))
    metadata_dict['removed_columns'] = ', '.join(columns_to_remove)
    
    # Convert to PyArrow metadata format
    metadata = {}
    for key, value in metadata_dict.items():
        metadata[key.encode()] = str(value).encode()
    
    # Save cleaned data with preserved metadata
    table = pa.Table.from_pandas(df_clean, preserve_index=False)
    
    # Add metadata to table
    existing_metadata = table.schema.metadata or {}
    existing_metadata.update(metadata)
    table = table.replace_schema_metadata(existing_metadata)
    
    # Write cleaned parquet file
    pq.write_table(table, parquet_file)
    
    print(f"✅ Cleaned file saved: {parquet_file}")
    print(f"📊 File size change: {parquet_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Verify the cleaned file
    print("\n🔍 Verifying cleaned file...")
    df_verify = pd.read_parquet(parquet_file)
    print(f"✅ Verification: {df_verify.shape[0]:,} rows, {df_verify.shape[1]} columns")
    
    # Show final column list
    print(f"\n📋 Final columns ({len(df_verify.columns)}):")
    for i, col in enumerate(df_verify.columns, 1):
        available = df_verify[col].notna().sum()
        missing_pct = (len(df_verify) - available) / len(df_verify) * 100
        print(f"  {i:2d}. {col:<20} - {available:,} values ({100-missing_pct:5.1f}% available)")

if __name__ == "__main__":
    cleanup_mmf9_parquet()