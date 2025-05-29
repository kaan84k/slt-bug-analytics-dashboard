import pandas as pd
import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (for OpenAI API key)
load_dotenv()

# Ensure API key is available
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found. Please create a .env file with your API key.")
    sys.exit(1)

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_bug_data(file_path='reclassified_bugs_with_sbert.csv'):
    """Load and prepare the bug data"""
    try:
        logger.info(f"Loading bug data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['review_description', 'bug_category', 'review_date', 'appVersion']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            sys.exit(1)
            
        return df
    except Exception as e:
        logger.error(f"Error loading bug data: {str(e)}")
        sys.exit(1)

def group_bugs_by_category(df):
    """Group bugs by category and return a dictionary"""
    logger.info("Grouping bugs by category")
    grouped = {}
    
    for category in df['bug_category'].unique():
        category_df = df[df['bug_category'] == category]
        grouped[category] = category_df
        logger.info(f"Category: {category} - {len(category_df)} bugs")
        
    return grouped

def generate_summary_with_gpt(category, reviews_df, max_reviews=25):
    """Generate summaries, key findings, suggested actions, and priority levels using GPT-3.5"""
    # Limit the number of reviews to avoid token limit issues
    if len(reviews_df) > max_reviews:
        logger.info(f"Limiting {category} reviews from {len(reviews_df)} to {max_reviews}")
        reviews_sample = reviews_df.sample(max_reviews, random_state=42)
    else:
        reviews_sample = reviews_df
    
    # Prepare input for GPT
    reviews_text = "\n\n".join([
        f"Review {i+1} (Date: {row['review_date']}, Version: {row['appVersion']}): {row['review_description']}"
        for i, (_, row) in enumerate(reviews_sample.iterrows())
    ])
    
    prompt = f"""
    You are a software development expert tasked with analyzing user reviews for bug reports related to: {category}.
    
    Please analyze the following user reviews and provide:
    
    1. SUMMARY: A concise summary of the bug issues reported (2-3 sentences).
    2. KEY FINDINGS: The main technical problems identified (bullet points).
    3. SUGGESTED ACTIONS: Specific technical fixes or investigations the development team should undertake (bullet points).
    4. PRIORITY LEVEL: Assign a priority level (Critical, High, Medium, Low) based on:
       - User impact
       - Frequency of reports
       - Technical severity
    
    Here are the user reviews:
    {reviews_text}
    
    Format your response with clear headers for each section.
    """
    
    # Call OpenAI API
    try:
        # Add a retry mechanism with exponential backoff
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Updated code for OpenAI API v1.0.0+
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a software development expert providing bug analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,  # More deterministic
                    max_tokens=1000
                )
                
                response = completion.choices[0].message.content
                logger.debug(f"GPT response for {category}: {response[:100]}...")
                return parse_gpt_response(response)
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API call failed, retrying in {retry_delay} seconds... ({str(e)})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
                    
    except Exception as e:
        logger.error(f"Error generating summary for {category}: {str(e)}")
        return {
            "summary": f"Error generating summary: {str(e)}",
            "key_findings": "Error",
            "suggested_actions": "Error",
            "priority_level": "Unknown",
            "additional_notes": f"Error occurred during analysis: {str(e)}"
        }

def parse_gpt_response(response):
    """Parse the GPT response into structured fields"""
    result = {
        "summary": "",
        "key_findings": "",
        "suggested_actions": "",
        "priority_level": "",
        "additional_notes": ""
    }
    
    # Split response by sections
    sections = response.split("\n\n")
    
    current_section = None
    for section in sections:
        section = section.strip()
        section_lower = section.lower()
        
        if "summary" in section_lower[:30]:
            current_section = "summary"
            # Clean up formatting and remove prefixes like '### 1.' or '****'
            clean_text = section.split("summary:", 1)[-1] if "summary:" in section.lower() else section
            clean_text = clean_text.strip()
            # Remove any markdown headers or numbering prefixes
            clean_text = clean_text.lstrip("*#1234567890. ")
            result["summary"] = clean_text
        
        elif "key findings" in section_lower[:30]:
            current_section = "key_findings"
            # Clean up the section header
            clean_text = section.split("key findings:", 1)[-1] if "key findings:" in section.lower() else section
            clean_text = clean_text.strip()
            clean_text = clean_text.lstrip("*#1234567890. ")
            result["key_findings"] = clean_text
        
        elif "suggested actions" in section_lower[:35]:
            current_section = "suggested_actions"
            # Clean up the section header
            clean_text = section.split("suggested actions:", 1)[-1] if "suggested actions:" in section.lower() else section
            clean_text = clean_text.strip()
            clean_text = clean_text.lstrip("*#1234567890. ")
            result["suggested_actions"] = clean_text
        
        elif "priority" in section_lower[:30] or "level" in section_lower[:30]:
            current_section = "priority_level"
            # Extract the full priority text
            priority_text = section.split("priority level:", 1)[-1] if "priority level:" in section.lower() else section
            priority_text = priority_text.strip()
            priority_text = priority_text.lstrip("*#1234567890. ")
            
            # Store the full priority analysis as additional notes
            result["additional_notes"] = priority_text
            
            # Extract just the priority level value
            if "critical" in priority_text.lower():
                result["priority_level"] = "Critical"
            elif "high" in priority_text.lower():
                result["priority_level"] = "High"
            elif "medium" in priority_text.lower():
                result["priority_level"] = "Medium"
            elif "low" in priority_text.lower():
                result["priority_level"] = "Low"
            else:
                result["priority_level"] = "Unknown"
                
        elif current_section and section:
            # Append additional content to current section
            # Clean up any formatting or prefixes
            clean_text = section.lstrip("*#1234567890. ")
            result[current_section] += "\n" + clean_text
    
    # Final cleanup to remove any markdown or numbering artifacts
    for key in result:
        if result[key]:
            # Remove markdown headers like "### 1." or "### 2." 
            result[key] = result[key].replace("### 1.", "").replace("### 2.", "")
            result[key] = result[key].replace("### 3.", "").replace("### 4.", "")
            # Remove asterisks formatting
            result[key] = result[key].replace("****", "")
    
    return result

def main():
    logger.info("Starting developer summary generation")
    
    # Load bug data
    bugs_df = load_bug_data()
    
    # Group bugs by category
    grouped_bugs = group_bugs_by_category(bugs_df)
    
    # Prepare the results dataframe
    results = []
    
    # Process each category
    for category, category_df in tqdm(grouped_bugs.items(), desc="Processing bug categories"):
        logger.info(f"Generating summary for category: {category}")
          # Generate summary with GPT
        summary_data = generate_summary_with_gpt(category, category_df)
        
        # Add to results
        results.append({
            "bug_category": category,
            "count": len(category_df),
            "summary": summary_data["summary"],
            "key_findings": summary_data["key_findings"],
            "suggested_actions": summary_data["suggested_actions"],
            "priority_level": summary_data["priority_level"],
            "additional_notes": summary_data["additional_notes"],
            "latest_date": category_df['review_date'].max(),
            "versions_affected": ", ".join(category_df['appVersion'].dropna().unique().astype(str))
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Sort by priority level
    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
        "Unknown": 4
    }
    
    results_df['priority_sort'] = results_df['priority_level'].map(priority_order)
    results_df.sort_values('priority_sort', inplace=True)
    results_df.drop('priority_sort', axis=1, inplace=True)
    
    # Save to CSV
    output_file = "developer_bug_summaries.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Developer summaries saved to {output_file}")
    
    print(f"\nDeveloper bug summaries generated successfully!")
    print(f"Output file: {output_file}")
    print(f"Total bug categories processed: {len(results_df)}")
    
    # Show summary of priority levels
    priority_counts = results_df['priority_level'].value_counts()
    print("\nPriority Summary:")
    for priority, count in priority_counts.items():
        print(f"  {priority}: {count} categories")

if __name__ == "__main__":
    main()