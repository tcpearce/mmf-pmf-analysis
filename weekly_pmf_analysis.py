#!/usr/bin/env python3
"""
Weekly PMF Source Apportionment Analysis
========================================

This script runs PMF source apportionment analysis for every complete week between
1st September 2023 (when data became reliable) to 1st March 2024.

Features:
- Automated weekly analysis for all MMF stations
- PDF conversion of HTML dashboards
- Organized output directory structure
- Progress tracking and error handling
- Comprehensive logging

Author: PMF Analysis System
Date: 2025-09-13
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Import for PDF conversion
try:
    import pdfkit
    HAS_PDFKIT = True
    print("pdfkit available for PDF conversion")
except ImportError:
    HAS_PDFKIT = False
    print("pdfkit not available")

# Skip weasyprint due to Windows library dependencies
HAS_WEASYPRINT = False
print("weasyprint disabled (Windows library issues)")

if not (HAS_PDFKIT or HAS_WEASYPRINT):
    print("Warning: No PDF conversion libraries available - will use Chrome/text fallback")

class WeeklyPMFAnalyzer:
    def __init__(self, start_date="2023-09-01", end_date="2024-03-01", 
                 output_base_dir="weekly_pmf_results", stations=None,
                 factors=None, models=20, run_pca=True, max_workers=2, 
                 run_full_period=True):
        """
        Initialize Weekly PMF Analyzer.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD) 
            output_base_dir (str): Base output directory
            stations (list): List of stations to analyze (default: all)
            factors (int): Number of factors (None for optimization)
            models (int): Number of models to run
            run_pca (bool): Whether to run PCA analysis
            max_workers (int): Maximum parallel processes
            run_full_period (bool): Whether to also run analysis for full period
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Validate date order
        if self.start_date >= self.end_date:
            raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
        self.output_base_dir = Path(output_base_dir)
        self.stations = stations or ['MMF1', 'MMF2', 'MMF6', 'MMF9']
        self.factors = factors
        self.models = models
        self.run_pca = run_pca
        self.max_workers = max_workers
        self.run_full_period = run_full_period
        
        # Create output directory structure
        self.output_base_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self.setup_logging()
        
        # Generate weekly date ranges
        self.weekly_ranges = self.generate_weekly_ranges()
        
        total_weekly = len(self.weekly_ranges) * len(self.stations)
        total_full_period = len(self.stations) if self.run_full_period else 0
        total_analyses = total_weekly + total_full_period
        
        self.logger.info(f"Initialized Weekly PMF Analyzer")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Stations: {self.stations}")
        self.logger.info(f"Weekly ranges: {len(self.weekly_ranges)} weeks")
        self.logger.info(f"Weekly analyses planned: {total_weekly}")
        if self.run_full_period:
            self.logger.info(f"Full period analyses planned: {total_full_period}")
        self.logger.info(f"Total analyses planned: {total_analyses}")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_base_dir / "weekly_pmf_analysis.log"
        
        # Create logger
        self.logger = logging.getLogger('WeeklyPMF')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def generate_weekly_ranges(self):
        """Generate complete weekly date ranges."""
        self.logger.info("Generating weekly date ranges...")
        
        weekly_ranges = []
        current_date = self.start_date
        
        # Find the first Monday on or after start_date
        days_to_monday = (7 - current_date.weekday()) % 7
        if days_to_monday == 0 and current_date.weekday() != 0:
            days_to_monday = 7
        first_monday = current_date + timedelta(days=days_to_monday)
        
        self.logger.info(f"First Monday: {first_monday.strftime('%Y-%m-%d')}")
        
        current_monday = first_monday
        
        while current_monday < self.end_date:
            week_end = current_monday + timedelta(days=6)  # Sunday
            
            # Only include complete weeks that don't exceed end_date
            if week_end < self.end_date:
                weekly_ranges.append({
                    'start': current_monday,
                    'end': week_end,
                    'week_str': f"{current_monday.strftime('%Y%m%d')}_{week_end.strftime('%Y%m%d')}",
                    'week_display': f"{current_monday.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
                })
            
            current_monday += timedelta(weeks=1)
        
        self.logger.info(f"Generated {len(weekly_ranges)} complete weekly ranges")
        for i, week in enumerate(weekly_ranges[:5]):  # Show first 5 weeks
            self.logger.info(f"  Week {i+1}: {week['week_display']}")
        if len(weekly_ranges) > 5:
            self.logger.info(f"  ... and {len(weekly_ranges) - 5} more weeks")
        
        return weekly_ranges
    
    def run_single_analysis(self, station, week_info):
        """Run PMF analysis for a single station and week."""
        start_date_str = week_info['start'].strftime('%Y-%m-%d')
        end_date_str = week_info['end'].strftime('%Y-%m-%d')
        week_str = week_info['week_str']
        
        # Create output directory for this analysis
        analysis_dir = self.output_base_dir / f"{station}_{week_str}"
        analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # Build command
        cmd = [
            'python', 'pmf_source_app.py',
            station,
            '--start-date', start_date_str,
            '--end-date', end_date_str,
            '--output-dir', str(analysis_dir),
            '--models', str(self.models)
        ]
        
        if self.factors:
            cmd.extend(['--factors', str(self.factors)])
        
        if self.run_pca:
            cmd.append('--run-pca')
        
        analysis_id = f"{station}_{week_str}"
        
        try:
            self.logger.info(f"Starting analysis: {analysis_id}")
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            start_time = time.time()
            
            # Run the analysis with proper encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '1'
            
            # Use bytes mode to avoid encoding issues, then decode manually
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=1800,  # 30 minute timeout
                env=env
            )
            
            # Safely decode output
            try:
                stdout_text = result.stdout.decode('utf-8', errors='replace')
                stderr_text = result.stderr.decode('utf-8', errors='replace')
            except AttributeError:
                # If already text, use as is
                stdout_text = result.stdout if result.stdout else ''
                stderr_text = result.stderr if result.stderr else ''
            
            # Create a result-like object with decoded text
            class DecodedResult:
                def __init__(self, original_result, stdout, stderr):
                    self.returncode = original_result.returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            result = DecodedResult(result, stdout_text, stderr_text)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.logger.info(f"Completed {analysis_id} in {duration:.1f}s")
                
                # Convert dashboard to PDF
                pdf_path = self.convert_dashboard_to_pdf(analysis_dir, station, week_str)
                
                return {
                    'status': 'success',
                    'analysis_id': analysis_id,
                    'station': station,
                    'week': week_info,
                    'duration': duration,
                    'output_dir': analysis_dir,
                    'pdf_path': pdf_path,
                    'stdout': result.stdout[-1000:] if result.stdout else '',  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
            else:
                self.logger.error(f"Failed {analysis_id}: return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                
                return {
                    'status': 'failed',
                    'analysis_id': analysis_id,
                    'station': station,
                    'week': week_info,
                    'duration': duration,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout {analysis_id} after 30 minutes")
            return {
                'status': 'timeout',
                'analysis_id': analysis_id,
                'station': station,
                'week': week_info,
                'duration': 1800
            }
        
        except Exception as e:
            self.logger.error(f"Exception in {analysis_id}: {e}")
            return {
                'status': 'exception',
                'analysis_id': analysis_id,
                'station': station,
                'week': week_info,
                'error': str(e)
            }
    
    def convert_dashboard_to_pdf(self, analysis_dir, station, week_str):
        """Convert HTML dashboard to PDF."""
        # Find the HTML dashboard file
        html_files = list(analysis_dir.glob("*_pmf_dashboard.html"))
        
        if not html_files:
            self.logger.warning(f"No HTML dashboard found in {analysis_dir}")
            return None
        
        html_file = html_files[0]
        pdf_file = analysis_dir / f"{station}_PMF_Dashboard_{week_str}.pdf"
        
        try:
            # Method 1: Use weasyprint (best option, pure Python)
            if HAS_WEASYPRINT:
                try:
                    weasyprint.HTML(filename=str(html_file)).write_pdf(str(pdf_file))
                    self.logger.info(f"PDF created with weasyprint: {pdf_file.name}")
                    return pdf_file
                except Exception as e:
                    self.logger.warning(f"Weasyprint failed: {e}, trying next method")
            
            # Method 2: Use pdfkit (requires wkhtmltopdf)
            if HAS_PDFKIT:
                try:
                    options = {
                        'page-size': 'A4',
                        'orientation': 'Portrait',
                        'margin-top': '0.75in',
                        'margin-right': '0.75in',
                        'margin-bottom': '0.75in',
                        'margin-left': '0.75in',
                        'encoding': "UTF-8",
                        'no-outline': None,
                        'enable-local-file-access': None
                    }
                    
                    pdfkit.from_file(str(html_file), str(pdf_file), options=options)
                    self.logger.info(f"PDF created with pdfkit: {pdf_file.name}")
                    return pdf_file
                except Exception as e:
                    # Only log pdfkit failures if it's not the common missing executable issue
                    if 'wkhtmltopdf executable found' not in str(e):
                        self.logger.warning(f"pdfkit failed: {e}, trying next method")
                    else:
                        self.logger.debug(f"pdfkit not available (expected): {e}")
            
            # Method 3: Try using Chrome/Edge headless
            success = self.convert_with_chrome(html_file, pdf_file)
            if success:
                self.logger.info(f"PDF created with Chrome: {pdf_file.name}")
                return pdf_file
            
            # Method 4: Create a simple text-based report as fallback
            self.create_text_report(analysis_dir, station, week_str)
            self.logger.warning(f"All PDF methods failed, created text report for {station}_{week_str}")
            return None
        
        except Exception as e:
            self.logger.error(f"PDF conversion failed for {station}_{week_str}: {e}")
            try:
                self.create_text_report(analysis_dir, station, week_str)
                self.logger.info(f"Created fallback text report for {station}_{week_str}")
            except Exception as e2:
                self.logger.error(f"Even text report failed for {station}_{week_str}: {e2}")
            return None
    
    def convert_with_chrome(self, html_file, pdf_file):
        """Try to convert HTML to PDF using Chrome headless."""
        try:
            # Try common Chrome/Edge locations (most likely first)
            chrome_paths = [
                r'C:\Program Files\Google\Chrome\Application\chrome.exe',  # Most common
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                'chrome',  # If in PATH
                'google-chrome',
                'chromium-browser'
            ]
            
            for chrome_path in chrome_paths:
                try:
                    # Check if chrome executable exists
                    if not os.path.exists(chrome_path) and chrome_path not in ['chrome', 'google-chrome', 'chromium-browser']:
                        continue
                    
                    # Convert file paths to proper format
                    pdf_path_str = str(pdf_file.absolute()).replace('\\', '/')
                    html_path_str = html_file.absolute().as_uri()
                    
                    cmd = [
                        chrome_path,
                        '--headless=new',
                        '--disable-gpu', 
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                        '--run-all-compositor-stages-before-draw',
                        '--virtual-time-budget=25000',
                        f'--print-to-pdf={pdf_path_str}',
                        html_path_str
                    ]
                    
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    env['PYTHONLEGACYWINDOWSFSENCODING'] = '1'
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        timeout=60, 
                        env=env,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if result.returncode == 0 and pdf_file.exists():
                        return True
                
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    continue
            
            return False
        
        except Exception:
            return False
    
    def create_text_report(self, analysis_dir, station, week_str):
        """Create a simple text-based summary report."""
        report_file = analysis_dir / f"{station}_Summary_Report_{week_str}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"{station} PMF Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Week: {week_str}\n")
            f.write(f"Station: {station}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Try to read and include summary statistics
            try:
                # Look for summary files
                summary_files = list(analysis_dir.glob("*_summary.txt"))
                factor_summaries = list(analysis_dir.glob("dashboard/*_factor_summary.txt"))
                
                if summary_files:
                    f.write("Analysis Summary:\n")
                    f.write("-" * 20 + "\n")
                    with open(summary_files[0], 'r') as sf:
                        f.write(sf.read())
                    f.write("\n\n")
                
                for factor_file in factor_summaries:
                    f.write(f"{factor_file.stem.replace('_', ' ').title()}:\n")
                    f.write("-" * 20 + "\n")
                    with open(factor_file, 'r') as ff:
                        f.write(ff.read())
                    f.write("\n\n")
            
            except Exception as e:
                f.write(f"Could not include detailed statistics: {e}\n")
            
            f.write(f"Dashboard files generated: {len(list(analysis_dir.glob('dashboard/*.png')))}\n")
            f.write(f"Output directory: {analysis_dir}\n")
    
    def run_task(self, task_type, station, week):
        """Run a single task (weekly or full period)."""
        if task_type == 'weekly':
            return self.run_single_analysis(station, week)
        elif task_type == 'full_period':
            return self.run_full_period_analysis(station)
    
    def run_full_period_analysis(self, station):
        """Run PMF analysis for the entire period for a single station."""
        start_date_str = self.start_date.strftime('%Y-%m-%d')
        end_date_str = self.end_date.strftime('%Y-%m-%d')
        period_str = f"{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"
        
        # Create output directory for full period analysis
        full_period_dir = self.output_base_dir / "full_period_analysis"
        full_period_dir.mkdir(exist_ok=True, parents=True)
        
        analysis_dir = full_period_dir / f"{station}_{period_str}"
        analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # Build command
        cmd = [
            'python', 'pmf_source_app.py',
            station,
            '--start-date', start_date_str,
            '--end-date', end_date_str,
            '--output-dir', str(analysis_dir),
            '--models', str(self.models)
        ]
        
        if self.factors:
            cmd.extend(['--factors', str(self.factors)])
        
        if self.run_pca:
            cmd.append('--run-pca')
        
        analysis_id = f"{station}_FULLPERIOD_{period_str}"
        
        try:
            self.logger.info(f"Starting full period analysis: {analysis_id}")
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            start_time = time.time()
            
            # Run the analysis with proper encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSFSENCODING'] = '1'
            
            # Use bytes mode to avoid encoding issues, then decode manually
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=3600,  # 60 minute timeout for full period
                env=env
            )
            
            # Safely decode output
            try:
                stdout_text = result.stdout.decode('utf-8', errors='replace')
                stderr_text = result.stderr.decode('utf-8', errors='replace')
            except AttributeError:
                # If already text, use as is
                stdout_text = result.stdout if result.stdout else ''
                stderr_text = result.stderr if result.stderr else ''
            
            # Create a result-like object with decoded text
            class DecodedResult:
                def __init__(self, original_result, stdout, stderr):
                    self.returncode = original_result.returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            result = DecodedResult(result, stdout_text, stderr_text)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.logger.info(f"Completed full period {analysis_id} in {duration:.1f}s")
                
                # Convert dashboard to PDF
                pdf_path = self.convert_dashboard_to_pdf(analysis_dir, station, f"FULLPERIOD_{period_str}")
                
                return {
                    'status': 'success',
                    'analysis_id': analysis_id,
                    'station': station,
                    'period': 'full',
                    'duration': duration,
                    'output_dir': analysis_dir,
                    'pdf_path': pdf_path,
                    'stdout': result.stdout[-1000:] if result.stdout else '',
                    'stderr': result.stderr[-1000:] if result.stderr else ''
                }
            else:
                self.logger.error(f"Failed full period {analysis_id}: return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                
                return {
                    'status': 'failed',
                    'analysis_id': analysis_id,
                    'station': station,
                    'period': 'full',
                    'duration': duration,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout full period {analysis_id} after 60 minutes")
            return {
                'status': 'timeout',
                'analysis_id': analysis_id,
                'station': station,
                'period': 'full',
                'duration': 3600
            }
        
        except Exception as e:
            self.logger.error(f"Exception in full period {analysis_id}: {e}")
            return {
                'status': 'exception',
                'analysis_id': analysis_id,
                'station': station,
                'period': 'full',
                'error': str(e)
            }
    
    def run_all_analyses(self):
        """Run all weekly analyses and full period analyses for all stations."""
        total_weekly = len(self.weekly_ranges) * len(self.stations)
        total_full_period = len(self.stations) if self.run_full_period else 0
        total_analyses = total_weekly + total_full_period
        
        self.logger.info(f"Starting PMF analysis campaign")
        self.logger.info(f"Weekly analyses to run: {total_weekly}")
        if self.run_full_period:
            self.logger.info(f"Full period analyses to run: {total_full_period}")
        self.logger.info(f"Total analyses to run: {total_analyses}")
        
        # Create all analysis tasks - weekly first, then full period
        tasks = []
        for week in self.weekly_ranges:
            for station in self.stations:
                tasks.append(('weekly', station, week))
        
        if self.run_full_period:
            for station in self.stations:
                tasks.append(('full_period', station, None))
        
        # Track results
        results = {
            'success': [],
            'failed': [],
            'timeout': [],
            'exception': []
        }
        
        start_time = time.time()
        
        if self.max_workers == 1:
            # Sequential processing
            for i, (task_type, station, week) in enumerate(tasks):
                self.logger.info(f"Progress: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%)")
                result = self.run_task(task_type, station, week)
                results[result['status']].append(result)
        
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self.run_task, task_type, station, week): (task_type, station, week)
                    for task_type, station, week in tasks
                }
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_task):
                    completed += 1
                    task_type, station, week = future_to_task[future]
                    
                    try:
                        result = future.result()
                        results[result['status']].append(result)
                        
                        # Progress update
                        progress = completed / len(tasks) * 100
                        self.logger.info(f"Progress: {completed}/{len(tasks)} ({progress:.1f}%)")
                        
                    except Exception as e:
                        self.logger.error(f"Task failed with exception: {e}")
                        analysis_id = f"{station}_FULLPERIOD" if task_type == 'full_period' else f"{station}_{week['week_str']}"
                        results['exception'].append({
                            'analysis_id': analysis_id,
                            'station': station,
                            'period': task_type,
                            'error': str(e)
                        })
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Generate final report
        self.generate_final_report(results, total_duration)
        
        return results
    
    def generate_final_report(self, results, total_duration):
        """Generate comprehensive final report."""
        report_file = self.output_base_dir / "PMF_Analysis_Campaign_Report.md"
        
        # Calculate statistics
        total_analyses = sum(len(results[status]) for status in results)
        success_rate = len(results['success']) / total_analyses * 100 if total_analyses > 0 else 0
        
        # Separate weekly and full period results
        weekly_success = [r for r in results['success'] if r.get('period') != 'full']
        full_period_success = [r for r in results['success'] if r.get('period') == 'full']
        
        with open(report_file, 'w') as f:
            f.write("# PMF Source Apportionment Analysis Campaign Report\n\n")
            
            f.write("## Analysis Overview\n")
            f.write(f"- **Period**: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"- **Stations**: {', '.join(self.stations)}\n")
            f.write(f"- **Weekly ranges**: {len(self.weekly_ranges)}\n")
            if self.run_full_period:
                f.write(f"- **Full period analyses**: {len(self.stations)}\n")
            f.write(f"- **Total analyses**: {total_analyses}\n")
            f.write(f"- **Total duration**: {total_duration/3600:.1f} hours\n")
            f.write(f"- **Success rate**: {success_rate:.1f}%\n\n")
            
            f.write("## Results Summary\n")
            f.write(f"- **Successful**: {len(results['success'])}\n")
            f.write(f"- **Failed**: {len(results['failed'])}\n")
            f.write(f"- **Timeout**: {len(results['timeout'])}\n")
            f.write(f"- **Exception**: {len(results['exception'])}\n\n")
            
            # Successful analyses
            if results['success']:
                f.write("## Successful Analyses\n\n")
                for result in results['success'][:10]:  # Show first 10
                    f.write(f"- {result['analysis_id']}: {result['duration']:.1f}s")
                    if result.get('pdf_path'):
                        f.write(" (PDF created)")
                    f.write("\n")
                
                if len(results['success']) > 10:
                    f.write(f"- ... and {len(results['success']) - 10} more successful analyses\n")
                f.write("\n")
            
            # Failed analyses
            if results['failed']:
                f.write("## Failed Analyses\n\n")
                for result in results['failed']:
                    f.write(f"- {result['analysis_id']}: Return code {result.get('return_code', 'unknown')}\n")
                    if result.get('stderr'):
                        f.write(f"  Error: {result['stderr'][:200]}...\n")
                f.write("\n")
            
            # Timeouts
            if results['timeout']:
                f.write("## Timeout Analyses\n\n")
                for result in results['timeout']:
                    f.write(f"- {result['analysis_id']}: Exceeded 30 minute limit\n")
                f.write("\n")
            
            # Exceptions
            if results['exception']:
                f.write("## Exception Analyses\n\n")
                for result in results['exception']:
                    f.write(f"- {result['analysis_id']}: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.output_base_dir}/\n")
            f.write("|-- weekly_pmf_analysis.log\n")
            f.write("|-- PMF_Analysis_Campaign_Report.md\n")
            
            # Weekly analyses
            if weekly_success:
                f.write("|-- Weekly Analyses:\n")
                for station in self.stations:
                    f.write(f"|   |-- {station}_YYYYMMDD_YYYYMMDD/\n")
                    f.write("|   |   |-- dashboard/\n")
                    f.write("|   |   |-- *_pmf_dashboard.html\n")
                    f.write("|   |   |-- *_PMF_Dashboard_*.pdf\n")
            
            # Full period analyses
            if self.run_full_period and full_period_success:
                f.write("|-- full_period_analysis/\n")
                for station in self.stations:
                    f.write(f"|   |-- {station}_YYYYMMDD_YYYYMMDD/\n")
                    f.write("|   |   |-- dashboard/\n")
                    f.write("|   |   |-- *_pmf_dashboard.html\n")
                    f.write("|   |   |-- *_PMF_Dashboard_*.pdf\n")
            
            f.write("```\n")
        
        self.logger.info(f"Final report generated: {report_file}")
        self.logger.info(f"Analysis campaign complete!")
        self.logger.info(f"Success rate: {success_rate:.1f}% ({len(results['success'])}/{total_analyses})")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Weekly PMF Source Apportionment Analysis')
    
    parser.add_argument('--start-date', default='2023-09-01',
                       help='Start date (YYYY-MM-DD, default: 2023-09-01)')
    parser.add_argument('--end-date', default='2024-03-01',
                       help='End date (YYYY-MM-DD, default: 2024-03-01)')
    parser.add_argument('--output-dir', default='weekly_pmf_results',
                       help='Output directory (default: weekly_pmf_results)')
    parser.add_argument('--stations', nargs='+', default=['MMF1', 'MMF2', 'MMF6', 'MMF9'],
                       help='Stations to analyze (default: all MMF stations)')
    parser.add_argument('--factors', type=int, default=None,
                       help='Number of factors (default: optimize automatically)')
    parser.add_argument('--models', type=int, default=20,
                       help='Number of models to run (default: 20)')
    parser.add_argument('--no-pca', action='store_true',
                       help='Skip PCA analysis (default: include PCA)')
    parser.add_argument('--no-full-period', action='store_true',
                       help='Skip full period analysis (default: include full period)')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum parallel processes (default: 2)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show analysis plan without running')
    
    args = parser.parse_args()
    
    print("Weekly PMF Source Apportionment Analysis")
    print("=" * 50)
    
    analyzer = WeeklyPMFAnalyzer(
        start_date=args.start_date,
        end_date=args.end_date,
        output_base_dir=args.output_dir,
        stations=args.stations,
        factors=args.factors,
        models=args.models,
        run_pca=not args.no_pca,
        max_workers=args.max_workers,
        run_full_period=not args.no_full_period
    )
    
    if args.dry_run:
        weekly_analyses = len(analyzer.weekly_ranges) * len(analyzer.stations)
        full_period_analyses = len(analyzer.stations) if analyzer.run_full_period else 0
        total_analyses = weekly_analyses + full_period_analyses
        
        print(f"\n📋 Analysis Plan (DRY RUN):")
        print(f"Weeks to analyze: {len(analyzer.weekly_ranges)}")
        print(f"Stations: {analyzer.stations}")
        print(f"Weekly analyses: {weekly_analyses}")
        if analyzer.run_full_period:
            print(f"Full period analyses: {full_period_analyses}")
        print(f"Total analyses: {total_analyses}")
        print(f"Estimated time (20min/analysis): {total_analyses * 20 / 60:.1f} hours")
        
        print("\nFirst 5 weeks:")
        for i, week in enumerate(analyzer.weekly_ranges[:5]):
            print(f"  Week {i+1}: {week['week_display']}")
        
        return
    
    # Run the analysis
    results = analyzer.run_all_analyses()
    
    return results

if __name__ == "__main__":
    results = main()