import os
import shutil
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_organization.log'),
        logging.StreamHandler()
    ]
)

class DataOrganizer:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.categories = {
            'monthly': self.base_dir / 'monthly_data',
            'interim': self.base_dir / 'interim_studies',
            'historical': self.base_dir / 'historical_data',
            'task_specific': self.base_dir / 'special_studies',
            'raw': self.base_dir / 'raw_data',
            'processed': self.base_dir / 'processed_data'
        }
        
    def create_directory_structure(self):
        """Create the directory structure for organizing data."""
        for dir_path in self.categories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_file_date(self, filename):
        """Extract date from filename."""
        # Common date patterns in filenames
        patterns = [
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[- ]?\d{4}',
            r'\d{4}[- ](?:0[1-9]|1[0-2])[- ](?:0[1-9]|[12]\d|3[01])',
            r'\d{2}[- ](?:0[1-9]|1[0-2])[- ]\d{4}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(0)
        return None

    def categorize_file(self, filepath):
        """Determine the category of a data file."""
        filename = filepath.name.lower()
        
        if 'mmf' in filename or 'monitoring' in filename:
            return 'monthly'
        elif 'interim' in filename or 'study' in filename:
            return 'interim'
        elif any(year in filename for year in ['2017', '2018', '2019']):
            return 'historical'
        elif 'diffusion' in filename or 'special' in filename or 'task' in filename:
            return 'task_specific'
        elif 'raw' in filename:
            return 'raw'
        elif 'processed' in filename or 'analysis' in filename:
            return 'processed'
        else:
            return 'raw'  # default to raw data

    def analyze_excel_file(self, filepath):
        """Analyze the content of an Excel file."""
        try:
            # Read just the first few rows to analyze structure
            df = pd.read_excel(filepath, nrows=5)
            columns = df.columns.tolist()
            
            logging.info(f"File: {filepath.name}")
            logging.info(f"Columns: {columns}")
            logging.info(f"Shape: {df.shape}")
            
            return {
                'columns': columns,
                'shape': df.shape
            }
        except Exception as e:
            logging.error(f"Error analyzing {filepath}: {str(e)}")
            return None

    def organize_files(self, source_dir="."):
        """Organize Excel files from source directory into categorized structure."""
        source_path = Path(source_dir)
        
        # Find all Excel files
        excel_files = list(source_path.glob("**/*.xls*"))  # catches both .xls and .xlsx
        
        if not excel_files:
            logging.warning(f"No Excel files found in {source_dir}")
            return
        
        for file_path in excel_files:
            try:
                # Determine category
                category = self.categorize_file(file_path)
                date_str = self.get_file_date(file_path.name)
                
                # Create year subdirectory if date found
                if date_str:
                    try:
                        # Try to extract year from date string
                        year = re.search(r'\d{4}', date_str).group(0)
                        target_dir = self.categories[category] / year
                    except:
                        target_dir = self.categories[category]
                else:
                    target_dir = self.categories[category]
                
                # Ensure target directory exists
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy file to new location
                target_path = target_dir / file_path.name
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)
                    logging.info(f"Copied {file_path.name} to {target_path}")
                    
                    # Analyze file content
                    analysis = self.analyze_excel_file(target_path)
                    if analysis:
                        logging.info(f"Analysis for {file_path.name}:")
                        logging.info(f"  Columns: {analysis['columns']}")
                        logging.info(f"  Shape: {analysis['shape']}")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")

def main():
    # Initialize organizer
    organizer = DataOrganizer()
    
    # Create directory structure
    organizer.create_directory_structure()
    
    # Organize files from current directory
    organizer.organize_files()
    
    logging.info("Data organization complete")

if __name__ == "__main__":
    main()
