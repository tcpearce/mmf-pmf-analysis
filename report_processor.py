import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path

class ReportProcessor:
    def __init__(self, base_dir: str = "reports"):
        self.base_dir = Path(base_dir)
        self.report_types = {
            'monthly': self.base_dir / 'monthly',
            'interim': self.base_dir / 'interim',
            'health_risk': self.base_dir / 'health_risk',
            'task_specific': self.base_dir / 'task_specific',
            'exceedance': self.base_dir / 'exceedance',
            'final': self.base_dir / 'final'
        }
        
    def extract_date_from_title(self, title: str) -> Optional[datetime]:
        """Extract date from report title."""
        # Look for month year pattern (e.g., "March 2025", "March-2025")
        month_year_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[- ](\d{4})'
        match = re.search(month_year_pattern, title, re.IGNORECASE)
        if match:
            try:
                return datetime.strptime(match.group(0), '%B %Y')
            except ValueError:
                try:
                    return datetime.strptime(match.group(0), '%b %Y')
                except ValueError:
                    return None
        return None

    def categorize_report(self, title: str) -> str:
        """Determine report category based on title."""
        title_lower = title.lower()
        if "health risk assessment" in title_lower:
            return 'health_risk'
        elif "monthly air quality" in title_lower:
            return 'monthly'
        elif "interim" in title_lower:
            return 'interim'
        elif "exceedance" in title_lower:
            return 'exceedance'
        elif "final" in title_lower or "study of ambient air quality" in title_lower:
            return 'final'
        else:
            return 'task_specific'

    def process_link(self, title: str, url: str) -> Dict:
        """Process a report link and return metadata."""
        category = self.categorize_report(title)
        date = self.extract_date_from_title(title)
        
        return {
            'title': title,
            'url': url,
            'category': category,
            'date': date,
            'target_dir': self.report_types[category]
        }

    def download_report(self, url: str, target_path: Path) -> bool:
        """Download a report from the given URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(response.content)
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    def organize_reports(self, reports: List[Dict]) -> None:
        """Organize reports into appropriate directories."""
        for report in reports:
            if report['date']:
                filename = f"{report['date'].strftime('%Y_%m')}_{report['title'].replace(' ', '_')}.pdf"
            else:
                filename = f"{report['title'].replace(' ', '_')}.pdf"
            
            target_path = report['target_dir'] / filename
            if not target_path.exists():
                self.download_report(report['url'], target_path)

def main():
    processor = ReportProcessor()
    # Example usage would go here
    # This would be where we process the actual links from the webpage

if __name__ == "__main__":
    main()
