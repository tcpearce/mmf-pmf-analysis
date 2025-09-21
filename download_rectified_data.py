import requests
from pathlib import Path
import logging
import pandas as pd
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_rectified.log'),
        logging.StreamHandler()
    ]
)

# Define the data files
MMF_FILES = {
    'MMF1': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/49894',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF1 - Mar 21 to Aug 23.xlsx',
        'period': 'March 2021 - August 2023'
    },
    'MMF2': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/49895',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF2 - Mar 21 to Aug 23.xlsx',
        'period': 'March 2021 - August 2023'
    },
    'MMF6': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/50937',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF6 - Mar 21 to June 23.xlsx',
        'period': 'March 2021 - June 2023'
    },
    'MMF9': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/49896',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF9 - Mar 21 to Aug 23.xlsx',
        'period': 'March 2021 - August 2023'
    }
}

def download_file(url, target_path, session):
    """Download a file from URL to target path."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        response = session.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logging.info(f"Successfully downloaded {target_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def analyze_mmf_file(filepath):
    """Analyze contents of downloaded MMF file."""
    try:
        df = pd.read_excel(filepath)
        
        # Basic file info
        info = {
            'filename': filepath.name,
            'size_mb': round(filepath.stat().st_size / (1024 * 1024), 2),
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist()
        }
        
        # Try to identify date range
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            info['date_columns'] = date_cols
            try:
                # Convert first date column found to datetime
                date_col = df[date_cols[0]]
                if not pd.api.types.is_datetime64_any_dtype(date_col):
                    date_col = pd.to_datetime(date_col)
                info['date_range'] = f"{date_col.min()} to {date_col.max()}"
            except:
                pass
        
        # Check for H2S and other key columns
        h2s_cols = [col for col in df.columns if 'h2s' in col.lower() or 'hydrogen' in col.lower()]
        if h2s_cols:
            info['h2s_columns'] = h2s_cols
            # Basic stats for H2S if numeric
            for col in h2s_cols:
                try:
                    stats = df[col].describe()
                    info[f'{col}_stats'] = {
                        'mean': round(stats['mean'], 3),
                        'min': round(stats['min'], 3),
                        'max': round(stats['max'], 3)
                    }
                except:
                    pass
        
        logging.info(f"\nAnalysis of {filepath.name}:")
        for key, value in info.items():
            logging.info(f"{key}: {value}")
            
        return info
        
    except Exception as e:
        logging.error(f"Error analyzing {filepath}: {str(e)}")
        return None

def create_summary_report(analyses, output_dir):
    """Create a summary report of all downloaded data."""
    summary_path = output_dir / 'data_summary.md'
    
    with open(summary_path, 'w') as f:
        f.write("# MMF Data Summary\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for station, analysis in analyses.items():
            if analysis:
                f.write(f"## {station}\n")
                f.write(f"Period: {MMF_FILES[station]['period']}\n")
                f.write(f"Filename: {analysis['filename']}\n")
                f.write(f"Size: {analysis['size_mb']} MB\n")
                f.write(f"Records: {analysis['rows']}\n")
                
                if 'date_range' in analysis:
                    f.write(f"Date Range: {analysis['date_range']}\n")
                
                if 'h2s_columns' in analysis:
                    f.write("\nH2S Measurements:\n")
                    for col in analysis['h2s_columns']:
                        f.write(f"- {col}\n")
                        if f'{col}_stats' in analysis:
                            stats = analysis[f'{col}_stats']
                            f.write(f"  - Mean: {stats['mean']}\n")
                            f.write(f"  - Min: {stats['min']}\n")
                            f.write(f"  - Max: {stats['max']}\n")
                
                f.write("\n")

def main():
    base_dir = Path('mmf_data')
    session = requests.Session()
    analyses = {}
    
    # Download and organize each file
    for station, info in MMF_FILES.items():
        try:
            # Create station directory if it doesn't exist
            station_dir = base_dir / station / 'processed'
            station_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up target path
            target_path = station_dir / info['filename']
            
            # Download file
            if download_file(info['url'], target_path, session):
                # Analyze downloaded file
                analysis = analyze_mmf_file(target_path)
                analyses[station] = analysis
                
                if analysis:
                    # Save detailed analysis to a text file alongside the data
                    analysis_path = station_dir / f"{info['filename']}_analysis.txt"
                    with open(analysis_path, 'w') as f:
                        f.write(f"Analysis of {info['filename']}\n")
                        f.write(f"Period: {info['period']}\n\n")
                        for key, value in analysis.items():
                            f.write(f"{key}: {value}\n")
                
        except Exception as e:
            logging.error(f"Error processing {station}: {str(e)}")
    
    # Create summary report
    create_summary_report(analyses, base_dir)
    logging.info("Download and organization complete")

if __name__ == "__main__":
    main()
