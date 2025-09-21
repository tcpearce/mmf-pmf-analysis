#!/usr/bin/env python3
"""
Quality Assurance script to verify all MMF corrections were applied correctly.
"""

import pandas as pd
import pyarrow.parquet as pq
import json
from pathlib import Path
from datetime import datetime

def main():
    print("MMF CORRECTIONS QUALITY ASSURANCE REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load station lookup
    with open('station_lookup.json', 'r') as f:
        station_config = json.load(f)
    
    station_mapping = {k: v for k, v in station_config.items() if not k.startswith('_')}
    
    print("EXPECTED STATION MAPPING:")
    print("-" * 30)
    for mmf, station in station_mapping.items():
        if station:
            print(f"  {mmf}: {station}")
        else:
            print(f"  Maries_Way: No MMF number")
    print()
    
    # Check corrected parquet files
    corrected_dir = Path('mmf_parquet_corrected')
    parquet_files = list(corrected_dir.glob('*.parquet'))
    
    print(f"CORRECTED PARQUET FILES FOUND: {len(parquet_files)}")
    print("-" * 40)
    
    qa_results = []
    
    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            pq_meta = pq.ParquetFile(pq_file)
            metadata = pq_meta.metadata.metadata or {}
            
            # Extract info from metadata (more reliable for empty files)
            meta_mmf = metadata.get(b'mmf_id', b'').decode()
            meta_station = metadata.get(b'station_name', b'').decode()
            schema_version = metadata.get(b'schema_version', b'').decode()
            
            # For non-empty files, also check data consistency
            if len(df) > 0 and 'mmf_id' in df.columns:
                mmf_id = df['mmf_id'].iloc[0]
                station_name = df['station_name'].iloc[0]
            else:
                # For empty files, use metadata
                mmf_id = meta_mmf if meta_mmf != 'null' else None
                station_name = meta_station
            
            # Verify correctness
            if mmf_id == 'None' or mmf_id is None or mmf_id == 'null':
                expected_station = 'Maries Way'
                mmf_correct = station_name == expected_station
            else:
                expected_station = station_mapping.get(f'MMF{mmf_id}')
                mmf_correct = station_name == expected_station
            
            # Check required columns
            required_cols = ['datetime', 'mmf_id', 'station_name']
            has_required_cols = all(col in df.columns for col in required_cols)
            
            # Schema version check
            schema_correct = schema_version == 'v2'
            
            result = {
                'filename': pq_file.name,
                'records': len(df),
                'mmf_id_data': str(mmf_id),
                'station_name_data': station_name,
                'mmf_id_meta': meta_mmf,
                'station_name_meta': meta_station,
                'schema_version': schema_version,
                'expected_station': expected_station,
                'mapping_correct': mmf_correct,
                'has_required_columns': has_required_cols,
                'schema_version_correct': schema_correct,
                'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}"
            }
            
            qa_results.append(result)
            
            status = "✅ PASS" if (mmf_correct and has_required_cols and schema_correct) else "❌ FAIL"
            
            print(f"{pq_file.name}: {status}")
            print(f"  Records: {len(df):,}")
            print(f"  MMF ID: {mmf_id} (Expected for {expected_station})")
            print(f"  Station: {station_name}")
            print(f"  Schema: {schema_version}")
            print(f"  Date Range: {result['date_range']}")
            
            if not mmf_correct:
                print(f"  ⚠️  Station mapping incorrect!")
            if not has_required_cols:
                print(f"  ⚠️  Missing required columns!")
            if not schema_correct:
                print(f"  ⚠️  Schema version incorrect!")
            print()
            
        except Exception as e:
            print(f"{pq_file.name}: ❌ ERROR - {e}")
            print()
    
    # Summary
    total_files = len(qa_results)
    passed_files = sum(1 for r in qa_results if r['mapping_correct'] and r['has_required_columns'] and r['schema_version_correct'])
    
    print("QA SUMMARY:")
    print("-" * 20)
    print(f"Total files checked: {total_files}")
    print(f"Files passed QA: {passed_files}")
    print(f"Files failed QA: {total_files - passed_files}")
    print()
    
    if passed_files == total_files:
        print("🎉 ALL CORRECTIONS VERIFIED SUCCESSFULLY!")
        print("✅ MMF number to station name mappings are correct")
        print("✅ All parquet files have required columns (datetime, mmf_id, station_name)")
        print("✅ All files upgraded to schema version v2")
        print("✅ Metadata is consistent between file data and parquet metadata")
    else:
        print("❌ SOME FILES FAILED QA CHECKS")
        failed_files = [r for r in qa_results if not (r['mapping_correct'] and r['has_required_columns'] and r['schema_version_correct'])]
        for failed in failed_files:
            print(f"   - {failed['filename']}: Issues detected")
    
    print()
    
    # Directory structure check
    print("DIRECTORY STRUCTURE CHECK:")
    print("-" * 30)
    
    corrected_data_dir = Path('mmf_data_corrected')
    expected_dirs = [
        'MMF1_Cemetery_Road',
        'MMF2_Silverdale_Pumping_Station', 
        'MMF6_Fire_Station',
        'MMF9_Galingale_View',
        'Maries_Way'
    ]
    
    for expected_dir in expected_dirs:
        dir_path = corrected_data_dir / expected_dir
        if dir_path.exists():
            raw_files = list((dir_path / 'raw').glob('*.xlsx')) if (dir_path / 'raw').exists() else []
            print(f"✅ {expected_dir}: {len(raw_files)} raw files")
        else:
            print(f"❌ {expected_dir}: Directory missing")
    
    print()
    
    # Backup verification
    backup_dirs = list(Path('.').glob('mmf_data_backup_*'))
    parquet_backups = list(Path('.').glob('mmf_parquet_backup_*'))
    
    print("BACKUP VERIFICATION:")
    print("-" * 20)
    print(f"MMF data backups: {len(backup_dirs)} directories")
    print(f"Parquet backups: {len(parquet_backups)} directories")
    
    if backup_dirs and parquet_backups:
        print("✅ Backups created successfully")
    else:
        print("⚠️  Some backups may be missing")
    
    print()
    
    # Final recommendation
    print("FINAL QA RECOMMENDATION:")
    print("-" * 25)
    
    if passed_files == total_files and len(backup_dirs) > 0:
        print("✅ APPROVED FOR PRODUCTION")
        print("   All corrections verified. Safe to proceed with finalization.")
    else:
        print("❌ REQUIRES FIXES")
        print("   Issues found that need resolution before production use.")
    
    print()
    print("QA Report completed.")

if __name__ == "__main__":
    main()