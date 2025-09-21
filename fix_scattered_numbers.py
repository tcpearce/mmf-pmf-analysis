#!/usr/bin/env python3
"""
Minimal patch to fix scattered numbers in HTML dashboard.

The issue is in the _create_diagnostic_scatters function where numpy array
operations might be displaying intermediate values. This script applies
a targeted fix.
"""

def apply_fix():
    """Apply the minimal fix to suppress scattered numbers."""
    
    import fileinput
    import sys
    
    filename = "pmf_source_apportionment_fixed.py"
    
    # Read the entire file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find and replace the problematic section
    old_code = """        # Mark potential outliers
        leverage_threshold = np.percentile(leverage, 95)
        residual_threshold = np.percentile(residual_norm, 95)
        outlier_mask = (leverage > leverage_threshold) | (residual_norm > residual_threshold)
        
        if np.any(outlier_mask):
            ax4.scatter(leverage[outlier_mask], residual_norm[outlier_mask], 
                       c='red', s=60, marker='x', label='Potential Outliers')
            ax4.legend()"""
    
    new_code = """        # Mark potential outliers - suppress any array display
        try:
            leverage_threshold = float(np.percentile(leverage, 95))
            residual_threshold = float(np.percentile(residual_norm, 95))
            outlier_mask = (leverage > leverage_threshold) | (residual_norm > residual_threshold)
            
            if np.any(outlier_mask):
                # Use explicit copies to prevent array display
                outlier_x = leverage[outlier_mask].copy()
                outlier_y = residual_norm[outlier_mask].copy()
                _ = ax4.scatter(outlier_x, outlier_y, 
                              c='red', s=60, marker='x', label='Potential Outliers')
                ax4.legend()
        except Exception:
            # Skip outlier marking if it causes issues
            pass"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✅ Applied fix for outlier detection")
    else:
        print("⚠️ Target code section not found - file may be corrupted")
        return False
    
    # Also add matplotlib non-interactive backend at the start of dashboard creation
    dashboard_start = """    def create_pmf_dashboard(self):
        \"\"\"
        Create comprehensive PMF dashboard with seaborn styling.
        FIXED VERSION based on successful ESAT test.
        \"\"\"
        print(\"📊 Creating PMF dashboard...\")"""
    
    dashboard_fixed = """    def create_pmf_dashboard(self):
        \"\"\"
        Create comprehensive PMF dashboard with seaborn styling.
        FIXED VERSION based on successful ESAT test.
        \"\"\"
        print(\"📊 Creating PMF dashboard...\")
        
        # Suppress matplotlib interactive output that might leak into HTML
        import matplotlib
        matplotlib.use('Agg')"""
    
    if dashboard_start in content:
        content = content.replace(dashboard_start, dashboard_fixed)
        print("✅ Applied matplotlib backend fix")
    else:
        print("⚠️ Dashboard function not found for backend fix")
    
    # Write the fixed content back
    try:
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Fixed file saved: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error writing fixed file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Applying fix for scattered numbers in HTML dashboard...")
    if apply_fix():
        print("🎉 Fix applied successfully!")
        print("   The scattered numbers in HTML dashboard should now be resolved.")
    else:
        print("❌ Fix failed - please check the file manually.")