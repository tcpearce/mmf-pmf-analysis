import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
import requests
from report_processor import ReportProcessor

def extract_report_links(html_content: str) -> List[Dict]:
    """Extract report links and titles from the webpage content."""
    # Create BeautifulSoup object from HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all links
    links = soup.find_all('a')
    
    report_links = []
    for link in links:
        title = link.get_text().strip()
        url = link.get('href')
        
        # Skip empty or navigation links
        if not title or not url or 'return to' in title.lower():
            continue
            
        report_links.append({
            'title': title,
            'url': url
        })
    
    return report_links

def main():
    # Initialize the report processor
    processor = ReportProcessor()
    
    # Base URL for the reports
    base_url = "https://engageenvironmentagency.uk.engagementhq.com/previous-air-quality-reports"
    
    try:
        # Fetch the webpage content
        response = requests.get(base_url)
        response.raise_for_status()
        
        # Extract links from the webpage
        report_links = extract_report_links(response.text)
        
        # Process each link
        processed_reports = []
        for link in report_links:
            processed_report = processor.process_link(link['title'], link['url'])
            processed_reports.append(processed_report)
        
        # Organize the reports
        processor.organize_reports(processed_reports)
        
        print(f"Successfully processed {len(processed_reports)} reports")
        
    except Exception as e:
        print(f"Error processing reports: {str(e)}")

if __name__ == "__main__":
    main()
