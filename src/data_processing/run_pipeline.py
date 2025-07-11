import subprocess
import time
import os
from pathlib import Path

def run_step(module_name, step_number, description, args=None):
    print(f"\n=== Step {step_number}: {description} ===")
    print(f"Running {module_name}...")

    root_dir = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root_dir / 'src')

    cmd = ['python', '-m', module_name]
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=root_dir,
        env=env
    )
    
    if result.returncode == 0:
        print(f"✓ {module_name} completed successfully")
    else:
        print(f"✗ Error running {module_name}")
        print("Error output:")
        print(result.stderr)
        raise Exception(f"Pipeline failed at step {step_number}")
    
    return result

def main():
    print("Starting SLT Bug Analysis Pipeline...")
    start_time = time.time()

    # Step 1: Scrape app reviews
    run_step('data_processing.sltmobitel_app_review', 1, "Scraping app reviews")
    
    # Step 2: Identify potential bugs
    run_step('data_processing.prioritized_bugs', 2, "Identifying potential bugs")
    
    # Step 3: Initial bug categorization
    run_step('data_processing.bug__categories', 3, "Initial bug categorization")
    
    # Step 4: Reclassify bugs using SBERT
    run_step('data_processing.reclassified_bugs_with_sbert', 4, "Reclassifying bugs with SBERT")
    
    # Step 5: Generate final bug analysis and insights
    run_step('data_processing.bug__categories_v2', 5, "Generating final bug analysis")
    
    # Step 6: Generate developer summaries
    dev_args = []
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found - running developer_summary in dry-run mode")
        dev_args.append('--dry-run')
    run_step('data_processing.developer_summary', 6, "Generating developer summaries", args=dev_args)

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
    # Use the repository-level data directory for the final output check
    # "parents[2]" points to the project root (../../ from this file)
    data_dir = Path(__file__).resolve().parents[2] / "data"

    missing_files = [str(data_dir / f) for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        print("\n⚠️ Warning: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("Some pipeline steps may have failed.")
    
    # Inform the user how to launch the dashboard manually
    print("\nTo view the interactive dashboard, run:")
    print("  streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main()
