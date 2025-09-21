import os
import re
from datetime import datetime
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin, urlparse
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_reports.log'),
        logging.StreamHandler()
    ]
)

class ReportDownloader:
    def __init__(self, base_url, base_dir="reports"):
        self.base_url = base_url
        self.base_dir = Path(base_dir)
        self.session = requests.Session()
        self.downloaded_files = set()
        
    def sanitize_filename(self, filename):
        """Clean filename to be valid for Windows."""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove trailing spaces and dots
        filename = filename.strip('. ')
        # Ensure filename isn't too long
        if len(filename) > 240:  # Leave room for path
            filename = filename[:240]
        return filename

    def get_report_date(self, title):
        """Extract date from report title."""
        date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        match = re.search(date_pattern, title)
        if match:
            try:
                return datetime.strptime(match.group(), '%B %Y')
            except ValueError:
                return None
        return None

    def get_report_type(self, title, url):
        """Determine report type from title and URL."""
        title_lower = title.lower()
        if "health risk assessment" in title_lower:
            return "health_risk"
        elif "monthly" in title_lower or re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', title):
            return "monthly"
        elif "interim" in title_lower:
            return "interim"
        elif "exceedance" in title_lower:
            return "exceedance"
        elif "diffusion tube" in title_lower:
            return "task_specific"
        elif "study of ambient air quality" in title_lower and any(year in title_lower for year in ['2017', '2018', '2019']):
            return "historical_studies"
        elif "final" in title_lower or "study of ambient air quality" in title_lower:
            return "final"
        return "other"

    def download_file(self, url, filename, subdir):
        """Download a file and save it to the appropriate directory."""
        target_dir = self.base_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure filename has .pdf extension if it's a PDF URL
        if url.lower().endswith('.pdf') and not filename.lower().endswith('.pdf'):
            filename += '.pdf'
            
        target_path = target_dir / filename
        
        # Skip if already downloaded
        if target_path in self.downloaded_files:
            logging.info(f"Skipping already downloaded file: {filename}")
            return
        
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type:
                logging.warning(f"Warning: {url} is not a PDF (Content-Type: {content_type})")
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.downloaded_files.add(target_path)
            logging.info(f"Successfully downloaded: {filename}")
            
        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")

    def extract_links_from_content(self, content):
        """Extract report links from the webpage content."""
        reports = []
        
        # Parse the HTML-like content
        for section in content:
            text = section.get('content', '')
            # Extract report titles and associated dates
            title_pattern = r'(?:Monthly Air Quality Report|Health Risk Assessment|Interim Report|Study of Ambient Air Quality|Diffusion Tube Report)[^.]*'
            matches = re.finditer(title_pattern, text, re.IGNORECASE)
            
            for match in matches:
                title = match.group(0).strip()
                date = self.get_report_date(title)
                report_type = self.get_report_type(title, "")
                
                if date:
                    filename = self.sanitize_filename(f"{date.strftime('%Y_%m')}_{title}")
                else:
                    filename = self.sanitize_filename(title)
                
                reports.append({
                    'title': title,
                    'date': date,
                    'type': report_type,
                    'filename': filename
                })
        
        return reports

    def download_reports(self, content):
        """Process the webpage content and download reports."""
        reports = self.extract_links_from_content(content)
        
        # Group reports by type
        for report in reports:
            if report.get('url'):  # Only if we have a URL
                self.download_file(
                    report['url'],
                    report['filename'],
                    report['type']
                )

        return len(reports)

def main():
    # Base URL for the reports
    base_url = "https://engageenvironmentagency.uk.engagementhq.com/previous-air-quality-reports"
    
    # Load webpage content from our JSON file
    try:
        with open('webpage_content.json', 'r') as f:
            webpage_content = json.load(f)
    except Exception as e:
        logging.error(f"Error loading webpage content: {str(e)}")
        return

    # Initialize downloader
    downloader = ReportDownloader(base_url)
    
    # Process the content
    num_reports = downloader.download_reports(webpage_content['external_context'])
    
    logging.info(f"Processed {num_reports} reports")

if __name__ == "__main__":
    main()
