import json
from datetime import datetime
import re
from typing import Dict, List, Optional
import csv
from pathlib import Path

def extract_date(title: str) -> Optional[str]:
    """Extract date from report title."""
    # Look for month year pattern (e.g., "March 2025", "March-2025")
    month_year_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[- ](\d{4})'
    match = re.search(month_year_pattern, title, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def categorize_report(title: str) -> str:
    """Determine report category based on title."""
    title_lower = title.lower()
    if "health risk assessment" in title_lower:
        return 'Health Risk Assessment'
    elif "monthly air quality" in title_lower:
        return 'Monthly Report'
    elif "interim" in title_lower:
        return 'Interim Report'
    elif "exceedance" in title_lower:
        return 'Exceedance Report'
    elif "final" in title_lower or "study of ambient air quality" in title_lower:
        return 'Final Report'
    elif "diffusion tube" in title_lower:
        return 'Task Specific Analysis'
    else:
        return 'Other'

def parse_webpage_content(content: List[Dict]) -> List[Dict]:
    """Parse the webpage content to extract report information."""
    reports = []
    seen_dates = set()  # To track unique report dates
    
    # Extract text content from all sections
    all_text = ""
    for section in content:
        all_text += section.get('content', '')
    
    # First find specific report titles
    specific_reports = [
        # Interim Reports
        "Study of Ambient Air Quality at Silverdale, Interim Report 1 – March to September 2021",
        "Study of Ambient Air Quality at Silverdale, Interim Report 2 - 4 March 21 to 31 May 2022",
        # Final Reports
        "Study of Ambient Air Quality at Silverdale Road 6 July 2017 to 14 February 2018 - Version 1",
        "Study of Ambient Air Quality at Silverdale Road 6 July 2017 to 14 February 2018 - Version 2",
        "Study of Ambient Air Quality at Silverdale 15 January 2019 and 25 June 2019 - Version 1",
        "Study of Ambient Air Quality at Silverdale 15 January 2019 and 25 June 2019 - Version 2",
        # Task Specific Reports
        "Diffusion Tube Report - 14 April 2022 - 14 July 2022",
        # Comparison Reports
        "Comparisons of Ambient Air Quality Data at MMF9 Galingale View, Silverdale"
    ]
    
    for report in specific_reports:
        if report in all_text:
            date = extract_date(report)
            category = categorize_report(report)
            reports.append({
                'title': report,
                'date': date,
                'category': category
            })
    
    # Then find monthly reports and health risk assessments
    # First look for explicitly marked reports
    monthly_pattern = r'(?:EA )?Monthly Air Quality (?:Monitoring )?Report[s]?[\s-]*(?:for )?([A-Za-z]+ \d{4})'
    health_pattern = r'(?:UKHSA )?Health Risk Assessment[s]?[\s-]*(?:for )?([A-Za-z]+ \d{4})'
    
    # Then look for date patterns that appear to be monthly reports
    date_pattern = r'(?<!\S)(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}(?!\S)'
    
    # Process explicit monthly and health reports
    for pattern in [monthly_pattern, health_pattern]:
        matches = re.finditer(pattern, all_text, re.IGNORECASE)
        for match in matches:
            date_str = match.group(1) if match.group(1) else None
            if date_str and date_str not in seen_dates:
                seen_dates.add(date_str)
                title = match.group(0).strip()
                category = 'Monthly Report' if 'Monthly' in title else 'Health Risk Assessment'
                reports.append({
                    'title': title,
                    'date': date_str,
                    'category': category
                })
    
    # Process standalone dates as monthly reports
    matches = re.finditer(date_pattern, all_text, re.IGNORECASE)
    for match in matches:
        date_str = match.group(0)
        if date_str and date_str not in seen_dates:
            seen_dates.add(date_str)
            reports.append({
                'title': f"Monthly Air Quality Report - {date_str}",
                'date': date_str,
                'category': 'Monthly Report'
            })
    
    return sorted(reports, key=lambda x: (x['category'], x['date'] if x['date'] else ''))

def save_catalog(reports: List[Dict], output_dir: str = '.') -> None:
    """Save the catalog in both CSV and JSON formats."""
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = Path(output_dir) / 'report_catalog.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'date', 'category'])
        writer.writeheader()
        writer.writerows(reports)
    
    # Save as JSON
    json_path = Path(output_dir) / 'report_catalog.json'
    with open(json_path, 'w') as f:
        json.dump(reports, f, indent=2)
    
    # Generate summary
    summary = {}
    for report in reports:
        category = report['category']
        if category not in summary:
            summary[category] = 0
        summary[category] += 1
    
    # Save summary
    summary_path = Path(output_dir) / 'catalog_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Report Catalog Summary\n")
        f.write("====================\n\n")
        f.write(f"Total Reports: {len(reports)}\n\n")
        f.write("Breakdown by Category:\n")
        for category, count in summary.items():
            f.write(f"{category}: {count} reports\n")

def main():
    # Load context from provided webpage content
    with open('webpage_content.json', 'r') as f:
        webpage_content = json.load(f)
    
    # Parse the content
    reports = parse_webpage_content(webpage_content['external_context'])
    
    # Save the catalog
    save_catalog(reports)
    
    print("Catalog generated successfully!")
    print(f"Found {len(reports)} unique reports")

if __name__ == "__main__":
    main()
