#!/usr/bin/env python3
"""
Plot MMF environmental data with separate subplots for each measurement.
Includes proper units, missing data visualization, and polar plot for wind direction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import warnings
from mmf_config import MMF_PARQUET_DIR, get_mmf_parquet_file, get_corrected_mmf_files
warnings.filterwarnings('ignore')

# Import the analyzer to get units
from analyze_parquet_data import ParquetAnalyzer

class MMFPlotter:
    def __init__(self, parquet_file):
        self.parquet_file = Path(parquet_file)
        self.df = None
        self.units = {}
        
    def load_data(self):
        """Load parquet data and extract units."""
        try:
            # Load the data using our analyzer
            analyzer = ParquetAnalyzer(self.parquet_file)
            if not analyzer.load_data():
                return False
            
            self.df = analyzer.df
            
            # Get stored units from metadata
            stored_units = analyzer.extract_units_from_metadata()
            
            # Get units for all columns (stored or inferred)
            for col in self.df.columns:
                if col in stored_units:
                    self.units[col] = stored_units[col]
                else:
                    self.units[col] = analyzer.get_suspected_units(col)
            
            print(f"Successfully loaded: {self.parquet_file.name}")
            print(f"Data shape: {self.df.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading {self.parquet_file}: {str(e)}")
            return False
    
    def filter_date_range(self, start_date=None, end_date=None):
        """Filter data by date range."""
        if 'datetime' not in self.df.columns:
            print("No datetime column found!")
            return False
        
        original_len = len(self.df)
        
        if start_date:
            self.df = self.df[self.df['datetime'] >= pd.to_datetime(start_date)]
        
        if end_date:
            self.df = self.df[self.df['datetime'] <= pd.to_datetime(end_date)]
        
        filtered_len = len(self.df)
        
        if filtered_len == 0:
            print("No data found in specified date range!")
            return False
        
        print(f"Filtered data: {filtered_len:,} records ({original_len - filtered_len:,} excluded)")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        return True
    
    def create_plots(self, station_name, start_date=None, end_date=None):
        """Create comprehensive plots for all data columns."""
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.style.use('default')
        
        # Define columns to plot (exclude metadata and empty columns)
        plot_columns = []
        for col in self.df.columns:
            # Skip metadata columns
            if col in ['datetime', 'gas_data_available', 'particle_data_available', 'date']:
                continue
            # Skip non-numeric identifier columns
            if col in ['mmf_id', 'station_name']:
                continue
            # Skip unnamed/empty columns (typically 100% missing)
            if col.startswith('Unnamed:') and self.df[col].isna().sum() / len(self.df) > 0.95:
                continue
            # Skip columns with very little data in the filtered period
            if self.df[col].notna().sum() == 0:
                continue
            # Only include numeric columns or known plottable columns
            if pd.api.types.is_numeric_dtype(self.df[col]) or col == 'WD':
                plot_columns.append(col)
        
        # Separate wind direction for polar plot
        has_wind_direction = 'WD' in plot_columns
        if has_wind_direction:
            plot_columns.remove('WD')
        
        # Calculate subplot arrangement
        n_plots = len(plot_columns) + (1 if has_wind_direction else 0)
        
        # Create figure with optimal subplot arrangement
        if n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        elif n_plots <= 9:
            rows, cols = 3, 3
        elif n_plots <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        # Adjust figure size based on number of plots
        fig_width = max(15, cols * 4)
        fig_height = max(10, rows * 3)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create main title with MMF number and station name
        mmf_id = self.df['mmf_id'].iloc[0] if 'mmf_id' in self.df.columns and len(self.df) > 0 else None
        actual_station = self.df['station_name'].iloc[0] if 'station_name' in self.df.columns and len(self.df) > 0 else station_name
        
        date_str = ""
        if start_date or end_date:
            date_str = f" ({start_date or 'start'} to {end_date or 'end'})"
        
        # Format title with MMF number and station name
        if mmf_id and str(mmf_id).lower() not in ['none', 'nan']:
            main_title = f'MMF{mmf_id} - {actual_station} Environmental Data{date_str}'
        else:
            main_title = f'{actual_station} Environmental Data{date_str}'
        
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
        
        plot_idx = 1
        
        # Plot wind direction first (polar plot)
        if has_wind_direction:
            ax = fig.add_subplot(rows, cols, plot_idx, projection='polar')
            self.plot_wind_direction(ax)
            plot_idx += 1
        
        # Plot all other columns
        for col in plot_columns:
            if plot_idx <= rows * cols:
                ax = fig.add_subplot(rows, cols, plot_idx)
                self.plot_column_data(ax, col)
                plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, rows * cols + 1):
            fig.delaxes(fig.add_subplot(rows, cols, i))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def plot_wind_direction(self, ax):
        """Create polar plot for wind direction."""
        if 'WD' not in self.df.columns:
            return
        
        # Get wind direction data
        wd_data = self.df['WD'].copy()
        dates = self.df['datetime']
        
        # Convert degrees to radians for polar plot
        valid_mask = wd_data.notna()
        valid_wd = wd_data[valid_mask]
        valid_dates = dates[valid_mask]
        
        if len(valid_wd) > 0:
            # Convert degrees to radians
            wd_radians = np.deg2rad(valid_wd)
            
            # Create time-based radius (normalize to 0-1)
            time_radius = np.linspace(0.1, 1, len(valid_wd))
            
            # Plot valid data
            scatter = ax.scatter(wd_radians, time_radius, 
                               c=range(len(valid_wd)), 
                               cmap='viridis', 
                               s=20, 
                               alpha=0.6)
            
            # Customize polar plot
            ax.set_title(f'WD ({self.units.get("WD", "degrees")})\nPolar Wind Direction', 
                        pad=20, fontweight='bold')
            ax.set_theta_direction(-1)  # Clockwise
            ax.set_theta_zero_location('N')  # North at top
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.set_ylabel('Time Progress', labelpad=30)
        
        # Show missing data count
        missing_count = wd_data.isna().sum()
        if missing_count > 0:
            ax.text(0.5, -0.15, f'Missing: {missing_count:,} points', 
                   transform=ax.transAxes, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    def plot_column_data(self, ax, column):
        """Plot data for a single column with missing data visualization."""
        if column not in self.df.columns:
            return
        
        data = self.df[column].copy()
        dates = self.df['datetime']
        
        # Check if data is numeric
        is_numeric = pd.api.types.is_numeric_dtype(data)
        
        # Skip non-numeric columns that are not useful for plotting
        if not is_numeric and column in ['mmf_id', 'station_name', 'date']:
            # Display info instead of plotting
            unique_values = data.dropna().unique()
            info_text = f"Non-numeric column\nUnique values: {len(unique_values)}"
            if len(unique_values) <= 5:
                info_text += f"\n{', '.join(map(str, unique_values))}"
            
            ax.text(0.5, 0.5, info_text, 
                   transform=ax.transAxes, 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7),
                   fontsize=10)
            ax.set_title(f'{column} (Non-numeric)', fontweight='bold', pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Plot valid data for numeric columns
        valid_mask = data.notna()
        valid_data = data[valid_mask]
        valid_dates = dates[valid_mask]
        
        if len(valid_data) > 0 and is_numeric:
            # Use seaborn style line plot
            ax.plot(valid_dates, valid_data, 
                   linewidth=1, 
                   alpha=0.8,
                   color=sns.color_palette("husl", len(self.df.columns))[list(self.df.columns).index(column) % 10])
        
        # Identify missing data (plot later after autoscaling)
        missing_dates = []
        if is_numeric:
            missing_mask = data.isna()
            missing_dates = dates[missing_mask]
        
        # Customize subplot
        unit = self.units.get(column, '')
        if unit:
            ylabel = f'{column}\n({unit})'
        else:
            ylabel = column
        
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.tick_params(axis='x', rotation=45)
        
        # Autoscale y-axis for atmospheric pressure columns
        if is_numeric and len(valid_data) > 0:
            pressure_columns = ['AMB_PRES', 'Pressure', 'PRESS', 'atmospheric_pressure', 'atm_pressure']
            if any(pressure_col.lower() in column.lower() for pressure_col in pressure_columns):
                # Set y-axis limits to actual data range with small padding
                y_min, y_max = valid_data.min(), valid_data.max()
                y_range = y_max - y_min
                padding = y_range * 0.05  # 5% padding
                ax.set_ylim(y_min - padding, y_max + padding)
        
        # Plot missing data as red points after autoscaling
        if is_numeric and len(missing_dates) > 0:
            # For pressure columns, place missing data at bottom of visible range
            pressure_columns = ['AMB_PRES', 'Pressure', 'PRESS', 'atmospheric_pressure', 'atm_pressure']
            is_pressure = any(pressure_col.lower() in column.lower() for pressure_col in pressure_columns)
            
            if is_pressure and len(valid_data) > 0:
                # Get current y-limits after autoscaling
                y_lim = ax.get_ylim()
                # Place missing data at the bottom of the visible range
                missing_y = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.02  # 2% from bottom
            else:
                # Default: place at y=0 for other columns
                missing_y = 0
            
            ax.scatter(missing_dates, 
                      np.full(len(missing_dates), missing_y), 
                      color='red', 
                      s=8, 
                      alpha=0.6,
                      marker='.',
                      label=f'Missing ({len(missing_dates):,})')
        
        # Add statistics text box for numeric data only
        if len(valid_data) > 0 and is_numeric:
            try:
                stats_text = f'Mean: {valid_data.mean():.2f}\nStd: {valid_data.std():.2f}\nCount: {len(valid_data):,}'
                
                # Add autoscaling note for pressure columns
                pressure_columns = ['AMB_PRES', 'Pressure', 'PRESS', 'atmospheric_pressure', 'atm_pressure']
                if any(pressure_col.lower() in column.lower() for pressure_col in pressure_columns):
                    y_min, y_max = valid_data.min(), valid_data.max()
                    stats_text += f'\nRange: {y_min:.1f}-{y_max:.1f} (autoscaled)'
                
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                       fontsize=8)
            except (ValueError, TypeError) as e:
                # If statistics calculation fails, show basic info
                stats_text = f'Count: {len(valid_data):,}\n(Non-numeric data)'
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
                       fontsize=8)
        
        # Add legend if there's missing data
        if is_numeric and len(missing_dates) > 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Set title
        ax.set_title(f'{column}', fontweight='bold', pad=10)
    
    def save_plot(self, fig, station_name, start_date=None, end_date=None, output_dir="plots"):
        """Save the plot to file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create filename
        date_suffix = ""
        if start_date and end_date:
            date_suffix = f"_{start_date}_to_{end_date}"
        elif start_date:
            date_suffix = f"_from_{start_date}"
        elif end_date:
            date_suffix = f"_to_{end_date}"
        
        filename = f"{station_name}_environmental_data{date_suffix}.png"
        filepath = output_path / filename
        
        # Save with high DPI
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"Plot saved: {filepath}")
        return filepath

def main():
    parser = argparse.ArgumentParser(description='Plot MMF environmental data (corrected dataset)')
    parser.add_argument('station', choices=['MMF1', 'MMF2', 'MMF6', 'MMF9', 'Maries_Way'], 
                       help='MMF station to plot')
    parser.add_argument('--start-date', type=str, 
                       help='Start date (YYYY-MM-DD format)', default=None)
    parser.add_argument('--end-date', type=str, 
                       help='End date (YYYY-MM-DD format)', default=None)
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--show', action='store_true', 
                       help='Display plot interactively (in addition to saving)')
    
    args = parser.parse_args()
    
    # Use corrected file path
    try:
        parquet_file = get_mmf_parquet_file(args.station)
    except Exception as e:
        print(f"Error determining file path for {args.station}: {e}")
        return
    
    if not parquet_file.exists():
        print(f"Error: File {parquet_file} not found!")
        print("Available corrected files:")
        corrected_files = get_corrected_mmf_files()
        for mmf_id, file_path in corrected_files.items():
            exists = "✅" if file_path.exists() else "❌"
            print(f"  {exists} {mmf_id}: {file_path.name}")
        return
    
    # Create plotter and load data
    plotter = MMFPlotter(parquet_file)
    
    if not plotter.load_data():
        return
    
    # Get MMF ID and station name from loaded data
    mmf_id = plotter.df['mmf_id'].iloc[0] if 'mmf_id' in plotter.df.columns and len(plotter.df) > 0 else 'N/A'
    station_name = plotter.df['station_name'].iloc[0] if 'station_name' in plotter.df.columns and len(plotter.df) > 0 else 'Unknown'
    
    # Display station information at the top
    print("\n" + "=" * 60)
    print(f"📊 MMF ENVIRONMENTAL DATA PLOTTING")
    print("=" * 60)
    if mmf_id and mmf_id != 'N/A' and str(mmf_id).lower() != 'none':
        print(f"🏭 Station: MMF{mmf_id} - {station_name}")
    else:
        print(f"🏭 Station: {station_name} (No MMF number)")
    print(f"📅 Data Period: {plotter.df['datetime'].min().strftime('%Y-%m-%d')} to {plotter.df['datetime'].max().strftime('%Y-%m-%d')}")
    print(f"📈 Total Records: {len(plotter.df):,}")
    print("=" * 60)
    
    # Filter date range if specified
    if args.start_date or args.end_date:
        if not plotter.filter_date_range(args.start_date, args.end_date):
            return
    
    print(f"\n🎨 Creating plots for {station_name}...")
    
    # Create plots
    fig = plotter.create_plots(args.station, args.start_date, args.end_date)
    
    # Save plot
    filepath = plotter.save_plot(fig, args.station, args.start_date, args.end_date, args.output_dir)
    
    # Show plot if requested
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"\n✅ Plotting complete!")
    print(f"📊 Plot saved to: {filepath}")
    
    # Count actual plotted columns (same logic as in create_plots)
    plot_columns = []
    for col in plotter.df.columns:
        if col in ['datetime', 'gas_data_available', 'particle_data_available', 'date']:
            continue
        if col.startswith('Unnamed:') and plotter.df[col].isna().sum() / len(plotter.df) > 0.95:
            continue
        if plotter.df[col].notna().sum() == 0:
            continue
        plot_columns.append(col)
    
    print(f"📈 Plotted {len(plot_columns)} data columns")

if __name__ == "__main__":
    main()
