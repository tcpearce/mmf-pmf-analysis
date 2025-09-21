import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_data_organization.log'),
        logging.StreamHandler()
    ]
)

class MMFDataOrganizer:
    def __init__(self, base_dir="mmf_data_corrected"):
        self.base_dir = Path(base_dir)
        self.mmf_dirs = {
            'MMF1': self.base_dir / 'MMF1',
            'MMF2': self.base_dir / 'MMF2',
            'MMF6': self.base_dir / 'MMF6',
            'MMF9': self.base_dir / 'MMF9',
            'combined': self.base_dir / 'combined_analysis'
        }
        self.date_ranges = {
            'MMF1': ('March 2021', 'August 2023'),
            'MMF2': ('March 2021', 'August 2023'),
            'MMF6': ('March 2021', 'June 2023'),
            'MMF9': ('March 2021', 'August 2023')
        }
        
    def create_directory_structure(self):
        """Create the directory structure for organizing MMF data."""
        for dir_path in self.mmf_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for raw and processed data
            (dir_path / 'raw').mkdir(exist_ok=True)
            (dir_path / 'processed').mkdir(exist_ok=True)
            
    def analyze_mmf_file(self, filepath):
        """Analyze the content of an MMF data file."""
        try:
            # Read the Excel file
            df = pd.read_excel(filepath)
            
            # Basic file info
            info = {
                'filename': filepath.name,
                'size': filepath.stat().st_size,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'date_range': (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
                'parameters': [col for col in df.columns if 'H2S' in col or 'hydrogen' in col.lower()]
            }
            
            logging.info(f"\nAnalysis for {filepath.name}:")
            logging.info(f"Dimensions: {info['shape']}")
            logging.info(f"Columns: {info['columns']}")
            logging.info(f"H2S Parameters: {info['parameters']}")
            if info['date_range']:
                logging.info(f"Date Range: {info['date_range']}")
            
            return info
            
        except Exception as e:
            logging.error(f"Error analyzing {filepath}: {str(e)}")
            return None

    def identify_mmf_station(self, filename):
        """Identify which MMF station the file belongs to."""
        filename_lower = filename.lower()
        for station in ['mmf1', 'mmf2', 'mmf6', 'mmf9']:
            if station in filename_lower:
                return station.upper()
        return None

    def organize_files(self, source_dir="."):
        """Organize MMF data files from source directory."""
        source_path = Path(source_dir)
        
        # Find all Excel files
        excel_files = list(source_path.glob("**/*.xls*"))
        
        if not excel_files:
            logging.warning(f"No Excel files found in {source_dir}")
            return
        
        # Process each file
        for file_path in excel_files:
            try:
                # Identify MMF station
                station = self.identify_mmf_station(file_path.name)
                if not station:
                    logging.warning(f"Could not identify MMF station for {file_path.name}")
                    continue
                
                # Determine if file contains raw or processed data
                is_raw = 'raw' in file_path.name.lower()
                subdir = 'raw' if is_raw else 'processed'
                
                # Copy to appropriate directory
                target_dir = self.mmf_dirs[station] / subdir
                target_path = target_dir / file_path.name
                
                if not target_path.exists():
                    # Analyze file before copying
                    analysis = self.analyze_mmf_file(file_path)
                    if analysis:
                        # Copy file
                        target_dir.mkdir(parents=True, exist_ok=True)
                        os.copy2(file_path, target_path)
                        logging.info(f"Copied {file_path.name} to {target_path}")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")

    def create_data_inventory(self):
        """Create an inventory of all MMF data files."""
        inventory = []
        
        for station, station_dir in self.mmf_dirs.items():
            if station != 'combined':  # Skip the combined analysis directory
                for data_type in ['raw', 'processed']:
                    data_dir = station_dir / data_type
                    if data_dir.exists():
                        files = list(data_dir.glob("*.xls*"))
                        for file in files:
                            analysis = self.analyze_mmf_file(file)
                            if analysis:
                                inventory.append({
                                    'station': station,
                                    'data_type': data_type,
                                    **analysis
                                })
        
        # Save inventory to Excel
        inventory_df = pd.DataFrame(inventory)
        inventory_df.to_excel(self.base_dir / 'data_inventory.xlsx', index=False)
        logging.info(f"Created data inventory at {self.base_dir / 'data_inventory.xlsx'}")

def main():
    # Initialize organizer
    organizer = MMFDataOrganizer()
    
    # Create directory structure
    organizer.create_directory_structure()
    
    # Organize files
    organizer.organize_files()
    
    # Create inventory
    organizer.create_data_inventory()
    
    logging.info("MMF data organization complete")

if __name__ == "__main__":
    main()
