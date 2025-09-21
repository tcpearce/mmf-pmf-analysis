#!/usr/bin/env python3
"""
MMF Data Validation Script
==========================

This script cross-checks the MMF site survey results against the official 
EA/UKHSA statements to identify any discrepancies or missing data.

Usage:
    python mmf_data_validation.py
"""

import pandas as pd
from datetime import datetime, date
from pathlib import Path

class MMFDataValidator:
    def __init__(self):
        """Initialize the MMF data validator with official specifications."""
        self.official_specs = self._load_official_specifications()
        self.survey_results = None
        
    def _load_official_specifications(self):
        """Load the official EA/UKHSA specifications for MMF sites."""
        specs = {
            'MMF Cemetery Road': {
                'former_name': 'MMF1',
                'expected_date_start': '2021-04-14',
                'expected_date_end': '2024-08-31',
                'expected_days': 1226,
                'status': 'Decommissioned 2024-09-02',
                'expected_pollutants': {
                    'H2S': True,
                    'CH4': True,
                    'NO2': False,
                    'SO2': True,
                    'PM10': True,
                    'PM2.5': True,
                    'BTEX': False  # Benzene, toluene, ethylbenzene, xylene
                },
                'notes': 'H2S data prior to 2023-08-31 not included in risk assessment'
            },
            'MMF Maries Way': {
                'former_name': 'N/A',
                'expected_date_start': '2024-09-02',
                'expected_date_end': '2025-07-31',
                'expected_days': 333,
                'status': 'Commissioned 2024-09-02',
                'expected_pollutants': {
                    'H2S': True,
                    'CH4': True,
                    'NO2': False,
                    'SO2': True,
                    'PM10': True,
                    'PM2.5': True,
                    'BTEX': False
                },
                'notes': 'New station replacing Cemetery Road'
            },
            'MMF Silverdale Pumping Station': {
                'former_name': 'MMF2',
                'expected_date_start': '2021-03-05',
                'expected_date_end': '2025-06-30',
                'expected_days': 1503,
                'status': 'Decommissioned 2024-01-08, Recommissioned 2024-03-16, Final decommissioning 2025-06-30',
                'expected_pollutants': {
                    'H2S': True,
                    'CH4': True,
                    'NO2': True,
                    'SO2': False,
                    'PM10': True,
                    'PM2.5': True,
                    'BTEX': True
                },
                'notes': 'Gap in data 2024-01-09 to 2024-03-15'
            },
            'MMF Fire Station': {
                'former_name': 'MMF6',
                'expected_date_start': '2021-04-24',
                'expected_date_end': '2023-06-27',
                'expected_days': 795,
                'status': 'Decommissioned 2023-06-27',
                'expected_pollutants': {
                    'H2S': True,
                    'CH4': True,
                    'NO2': False,
                    'SO2': True,
                    'PM10': True,
                    'PM2.5': True,
                    'BTEX': False
                },
                'notes': 'Early decommissioning'
            },
            'MMF Galingale View': {
                'former_name': 'MMF9',
                'expected_date_start': '2021-03-06',
                'expected_date_end': '2025-07-31',
                'expected_days': 1604,
                'status': 'Active',
                'expected_pollutants': {
                    'H2S': True,
                    'CH4': True,
                    'NO2': True,
                    'SO2': True,
                    'PM10': True,
                    'PM2.5': True,
                    'BTEX': True
                },
                'notes': 'Most comprehensive monitoring'
            }
        }
        return specs
    
    def load_survey_results(self, csv_path="mmf_survey_results/mmf_site_summary.csv"):
        """Load the survey results from CSV."""
        self.survey_results = pd.read_csv(csv_path)
        print(f"✅ Loaded survey results for {len(self.survey_results)} sites")
    
    def validate_all_sites(self):
        """Perform comprehensive validation of all sites."""
        print("\n🔍 MMF DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        validation_results = {}
        
        for site_name, official_spec in self.official_specs.items():
            print(f"🏭 VALIDATING: {site_name}")
            print("-" * 50)
            
            # Find corresponding survey result
            survey_row = self._find_survey_result(site_name, official_spec['former_name'])
            
            if survey_row is None:
                print(f"❌ CRITICAL: No survey data found for {site_name}")
                validation_results[site_name] = {'status': 'MISSING_DATA', 'issues': ['No survey data found']}
                print()
                continue
            
            # Validate this site
            site_validation = self._validate_site(site_name, official_spec, survey_row)
            validation_results[site_name] = site_validation
            
            print()
        
        # Generate summary
        self._generate_validation_summary(validation_results)
        
        return validation_results
    
    def _find_survey_result(self, site_name, former_name):
        """Find the survey result for a given site."""
        # Try by display name first
        mask = self.survey_results['Display_Name'].str.contains(site_name, na=False)
        if mask.any():
            return self.survey_results[mask].iloc[0]
        
        # Try by site name
        if site_name == 'MMF Maries Way':
            mask = self.survey_results['Site_Code'] == 'Maries_Way'
            if mask.any():
                return self.survey_results[mask].iloc[0]
        
        # Try by former name (MMF code)
        if former_name != 'N/A':
            mask = self.survey_results['Site_Code'] == former_name
            if mask.any():
                return self.survey_results[mask].iloc[0]
        
        return None
    
    def _validate_site(self, site_name, official_spec, survey_row):
        """Validate a single site against official specifications."""
        validation = {
            'status': 'PASS',
            'issues': [],
            'warnings': [],
            'info': []
        }
        
        # Check date ranges
        expected_start = pd.to_datetime(official_spec['expected_date_start'])
        expected_end = pd.to_datetime(official_spec['expected_date_end'])
        expected_days = official_spec['expected_days']
        
        if pd.notna(survey_row['Date_Range_Start']) and survey_row['Date_Range_Start']:
            actual_start = pd.to_datetime(survey_row['Date_Range_Start'])
            actual_end = pd.to_datetime(survey_row['Date_Range_End']) if pd.notna(survey_row['Date_Range_End']) else None
            actual_days = survey_row['Date_Range_Days']
            
            # Validate start date (allow 1-2 days tolerance)
            start_diff = abs((actual_start - expected_start).days)
            if start_diff > 2:
                validation['issues'].append(f"Start date mismatch: Expected {expected_start.date()}, Got {actual_start.date()} ({start_diff} days difference)")
                validation['status'] = 'FAIL'
            elif start_diff > 0:
                validation['warnings'].append(f"Start date slightly off: Expected {expected_start.date()}, Got {actual_start.date()} ({start_diff} days difference)")
            else:
                validation['info'].append(f"✅ Start date matches: {actual_start.date()}")
            
            # Validate end date
            if actual_end:
                end_diff = abs((actual_end - expected_end).days)
                if end_diff > 2:
                    validation['issues'].append(f"End date mismatch: Expected {expected_end.date()}, Got {actual_end.date()} ({end_diff} days difference)")
                    validation['status'] = 'FAIL'
                elif end_diff > 0:
                    validation['warnings'].append(f"End date slightly off: Expected {expected_end.date()}, Got {actual_end.date()} ({end_diff} days difference)")
                else:
                    validation['info'].append(f"✅ End date matches: {actual_end.date()}")
            
            # Validate duration
            days_diff = abs(actual_days - expected_days)
            if days_diff > 5:
                validation['issues'].append(f"Duration mismatch: Expected {expected_days} days, Got {actual_days} days ({days_diff} days difference)")
                validation['status'] = 'FAIL'
            elif days_diff > 0:
                validation['warnings'].append(f"Duration slightly off: Expected {expected_days} days, Got {actual_days} days ({days_diff} days difference)")
            else:
                validation['info'].append(f"✅ Duration matches: {actual_days} days")
                
        else:
            validation['issues'].append("No date range data available in survey")
            validation['status'] = 'FAIL'
        
        # Check data availability
        if survey_row['Total_Records'] == 0:
            validation['issues'].append("No data records found")
            validation['status'] = 'FAIL'
        else:
            validation['info'].append(f"✅ Contains {survey_row['Total_Records']:,} records")
        
        # Validate pollutants
        self._validate_pollutants(site_name, official_spec, survey_row, validation)
        
        # Print validation results
        self._print_site_validation(site_name, official_spec, survey_row, validation)
        
        return validation
    
    def _validate_pollutants(self, site_name, official_spec, survey_row, validation):
        """Validate pollutant availability against expectations."""
        expected_pollutants = official_spec['expected_pollutants']
        
        # Get available columns from survey
        gas_columns = survey_row['Gas_Columns'].split('; ') if pd.notna(survey_row['Gas_Columns']) and survey_row['Gas_Columns'] else []
        pm_columns = survey_row['Particle_Columns'].split('; ') if pd.notna(survey_row['Particle_Columns']) and survey_row['Particle_Columns'] else []
        all_columns = survey_row['All_Columns'].split('; ') if pd.notna(survey_row['All_Columns']) and survey_row['All_Columns'] else []
        
        # Check each expected pollutant
        pollutant_checks = {
            'H2S': any('H2S' in col for col in gas_columns),
            'CH4': any('CH4' in col for col in gas_columns),
            'NO2': any('NO2' in col for col in gas_columns),
            'SO2': any('SO2' in col for col in gas_columns),
            'PM10': any('PM10' in col for col in pm_columns),
            'PM2.5': any('PM2.5' in col for col in pm_columns),
            'BTEX': any(btex in ' '.join(all_columns) for btex in ['benzene', 'toluene', 'ethylbenzene', 'xylene'])
        }
        
        for pollutant, expected in expected_pollutants.items():
            available = pollutant_checks[pollutant]
            
            if expected and not available:
                validation['issues'].append(f"Missing expected pollutant: {pollutant}")
                validation['status'] = 'FAIL'
            elif not expected and available:
                validation['warnings'].append(f"Unexpected pollutant found: {pollutant}")
            elif expected and available:
                validation['info'].append(f"✅ Expected pollutant available: {pollutant}")
        
        # Check for NOX (which includes NO2)
        if 'NOX' in ' '.join(gas_columns):
            validation['info'].append("ℹ️ NOX available (includes NO2)")
    
    def _print_site_validation(self, site_name, official_spec, survey_row, validation):
        """Print validation results for a site."""
        print(f"Former Name: {official_spec['former_name']}")
        print(f"Status: {official_spec['status']}")
        print(f"Survey Records: {survey_row['Total_Records']:,}")
        print(f"Data Completeness: {survey_row['Data_Completeness_Percent']}%")
        print()
        
        if validation['info']:
            print("✅ VALIDATION PASSED:")
            for info in validation['info']:
                print(f"   {info}")
            print()
        
        if validation['warnings']:
            print("⚠️ WARNINGS:")
            for warning in validation['warnings']:
                print(f"   {warning}")
            print()
        
        if validation['issues']:
            print("❌ ISSUES FOUND:")
            for issue in validation['issues']:
                print(f"   {issue}")
            print()
        
        print(f"Overall Status: {'✅ PASS' if validation['status'] == 'PASS' else '❌ FAIL'}")
    
    def _generate_validation_summary(self, validation_results):
        """Generate overall validation summary."""
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 40)
        
        passed = sum(1 for v in validation_results.values() if v['status'] == 'PASS')
        failed = sum(1 for v in validation_results.values() if v['status'] == 'FAIL')
        total = len(validation_results)
        
        print(f"Total Sites Validated: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        # List major issues
        major_issues = []
        for site_name, validation in validation_results.items():
            if validation['status'] == 'FAIL':
                major_issues.extend([f"{site_name}: {issue}" for issue in validation['issues']])
        
        if major_issues:
            print("🚨 MAJOR ISSUES REQUIRING ATTENTION:")
            for issue in major_issues:
                print(f"   • {issue}")
        else:
            print("🎉 All sites passed validation!")
        
        print()
        
        # Pollutant availability summary
        self._generate_pollutant_summary()
    
    def _generate_pollutant_summary(self):
        """Generate pollutant availability summary."""
        print("🧪 POLLUTANT AVAILABILITY SUMMARY")
        print("-" * 35)
        
        pollutant_matrix = []
        
        for site_name, spec in self.official_specs.items():
            survey_row = self._find_survey_result(site_name, spec['former_name'])
            if survey_row is None:
                continue
                
            gas_columns = survey_row['Gas_Columns'].split('; ') if pd.notna(survey_row['Gas_Columns']) and survey_row['Gas_Columns'] else []
            pm_columns = survey_row['Particle_Columns'].split('; ') if pd.notna(survey_row['Particle_Columns']) and survey_row['Particle_Columns'] else []
            all_columns = survey_row['All_Columns'].split('; ') if pd.notna(survey_row['All_Columns']) else []
            
            row = {
                'Site': site_name.replace('MMF ', ''),
                'H2S': '✓' if any('H2S' in col for col in gas_columns) else '✗',
                'CH4': '✓' if any('CH4' in col for col in gas_columns) else '✗',
                'NO2': '✓' if any('NO2' in col for col in gas_columns) or any('NOX' in col for col in gas_columns) else '✗',
                'SO2': '✓' if any('SO2' in col for col in gas_columns) else '✗',
                'PM10': '✓' if any('PM10' in col for col in pm_columns) else '✗',
                'PM2.5': '✓' if any('PM2.5' in col for col in pm_columns) else '✗',
                'BTEX': '✓' if any(btex in ' '.join(all_columns).lower() for btex in ['benzene', 'toluene', 'ethylbenzene', 'xylene']) else '✗',
                'Records': f"{survey_row['Total_Records']:,}" if survey_row['Total_Records'] > 0 else "No Data"
            }
            pollutant_matrix.append(row)
        
        # Print as table
        df_matrix = pd.DataFrame(pollutant_matrix)
        print(df_matrix.to_string(index=False))

def main():
    """Main function to run MMF data validation."""
    validator = MMFDataValidator()
    
    # Check if survey results exist
    survey_csv = Path("mmf_survey_results/mmf_site_summary.csv")
    if not survey_csv.exists():
        print("❌ Survey results not found. Please run mmf_site_survey.py first.")
        return
    
    # Load survey results and validate
    validator.load_survey_results(str(survey_csv))
    validation_results = validator.validate_all_sites()
    
    # Save validation report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"mmf_validation_report_{timestamp}.txt"
    
    print(f"\n💾 Validation complete! Results also available in console output above.")

if __name__ == "__main__":
    main()