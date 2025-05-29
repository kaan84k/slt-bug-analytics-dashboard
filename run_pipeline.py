import subprocess
import time
import os

def run_step(script_name, step_number, description):
    print(f"\n=== Step {step_number}: {description} ===")
    print(f"Running {script_name}...")
    
    result = subprocess.run(['python', script_name], 
                          capture_output=True, 
                          text=True)
    
    if result.returncode == 0:
        print(f"✓ {script_name} completed successfully")
    else:
        print(f"✗ Error running {script_name}")
        print("Error output:")
        print(result.stderr)
        raise Exception(f"Pipeline failed at step {step_number}")
    
    return result

def main():
    print("Starting SLT Bug Analysis Pipeline...")
    start_time = time.time()

    # Step 1: Scrape app reviews
    run_step('sltmobitel_app_review.py', 1, "Scraping app reviews")
    
    # Step 2: Identify potential bugs
    run_step('prioritized_bugs.py', 2, "Identifying potential bugs")
    
    # Step 3: Initial bug categorization
    run_step('bug__categories.py', 3, "Initial bug categorization")
    
    # Step 4: Reclassify bugs using SBERT
    run_step('reclassified_bugs_with_sbert.py', 4, "Reclassifying bugs with SBERT")
    
    # Step 5: Generate final bug analysis and insights
    run_step('bug__categories_v2.py', 5, "Generating final bug analysis")
    
    # Step 6: Generate developer summaries
    run_step('developer_summary.py', 6, "Generating developer summaries")

    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✓ Pipeline completed in {duration:.2f} seconds")
    print("\nOutput files generated:")
    print("- slt_selfcare_google_reviews.csv (Raw app reviews)")
    print("- prioritized_bugs.csv (Identified bug reports)")
    print("- categorized_bugs.csv (Initially categorized bugs)")
    print("- reclassified_bugs_with_sbert.csv (Final bug classification)")
    print("- developer_bug_summaries.csv (Developer insights)")
      # Check if all required files exist
    required_files = [
        'slt_selfcare_google_reviews.csv',
        'prioritized_bugs.csv',
        'categorized_bugs.csv',
        'reclassified_bugs_with_sbert.csv',
        'developer_bug_summaries.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n⚠️ Warning: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("Some pipeline steps may have failed.")
    
    # Launch the dashboard
    print("\nLaunching the Streamlit dashboard...")
    print("Please wait while the dashboard initializes...")
    
    # Use subprocess.run to wait for streamlit to start
    try:
        subprocess.run(
            ['streamlit', 'run', 'app.py'],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("Error launching dashboard:", e)
    except KeyboardInterrupt:
        print("\nDashboard closed by user.")

if __name__ == "__main__":
    main()
