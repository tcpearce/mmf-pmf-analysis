import requests
from pathlib import Path
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_download.log'),
        logging.StreamHandler()
    ]
)

# Define the data files
MMF_FILES = {
    'MMF1': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/49894/download',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF1 - Mar 21 to Aug 23.xlsx',
        'period': 'March 2021 - August 2023'
    },
    'MMF2': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/49895/download',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF2 - Mar 21 to Aug 23.xlsx',
        'period': 'March 2021 - August 2023'
    },
    'MMF6': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/50937/download',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF6 - Mar 21 to June 23.xlsx',
        'period': 'March 2021 - June 2023'
    },
    'MMF9': {
        'url': 'https://engageenvironmentagency.uk.engagementhq.com/22781/widgets/82020/documents/49896/download',
        'filename': 'Silverdale Ambient Air Monitoring Data - MMF9 - Mar 21 to Aug 23.xlsx',
        'period': 'March 2021 - August 2023'
    }
}

def download_file(url, target_path, session):
    """Download a file from URL to target path."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Get the actual file
        response = session.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Check if we got an Excel file
        content_type = response.headers.get('content-type', '').lower()
        if 'excel' not in content_type and 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' not in content_type:
            logging.warning(f"Response might not be an Excel file. Content-Type: {content_type}")
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logging.info(f"Successfully downloaded {target_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    base_dir = Path('mmf_data_corrected')
    session = requests.Session()
    
    # Download and organize each file
    for station, info in MMF_FILES.items():
        try:
            # Create station directory if it doesn't exist
            station_dir = base_dir / station / 'processed'
            station_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up target path
            target_path = station_dir / info['filename']
            
            # Remove existing file if it exists
            if target_path.exists():
                target_path.unlink()
            
            # Download file
            if download_file(info['url'], target_path, session):
                logging.info(f"Successfully processed {station}")
                
                # Verify file size
                if target_path.stat().st_size > 0:
                    logging.info(f"File size: {target_path.stat().st_size / (1024*1024):.2f} MB")
                else:
                    logging.warning(f"File {info['filename']} appears to be empty")
                
        except Exception as e:
            logging.error(f"Error processing {station}: {str(e)}")
    
    logging.info("Download complete")

if __name__ == "__main__":
    main()
