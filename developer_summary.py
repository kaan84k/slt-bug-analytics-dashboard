import pandas as pd
import os
import time
import logging
from openai import AsyncOpenAI # Changed import
from dotenv import load_dotenv
# from tqdm import tqdm # tqdm might be removed or replaced for async
import sys
import asyncio # Added asyncio
import json # Added for caching
from datetime import datetime # Added for cache timestamp

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

# Configure OpenAI client (Async)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Caching Setup ---
CACHE_FILE_PATH = "gpt_summary_cache.json"

def generate_review_hash(reviews_df):
    """Generates a simple hash based on the number of reviews."""
    # For simplicity, using review count. A more robust hash could involve content.
    return str(len(reviews_df))

def load_cache():
    """Loads the cache from a JSON file."""
    if os.path.exists(CACHE_FILE_PATH):
        try:
            with open(CACHE_FILE_PATH, 'r') as f:
                logger.info(f"Loading cache from {CACHE_FILE_PATH}")
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading cache file {CACHE_FILE_PATH}: {e}. Starting with an empty cache.")
            return {}
    return {}

def save_cache(cache_data):
    """Saves the cache to a JSON file."""
    try:
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(cache_data, f, indent=4)
        logger.info(f"Cache saved to {CACHE_FILE_PATH}")
    except IOError as e:
        logger.error(f"Error saving cache to {CACHE_FILE_PATH}: {e}")

def is_cache_valid(cached_item, current_reviews_df):
    """Checks if the cached item is still valid based on review count hash."""
    if not cached_item:
        return False
    expected_hash = generate_review_hash(current_reviews_df)
    return cached_item.get("review_count_hash") == expected_hash

# --- End Caching Setup ---

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
        # logger.info(f"Category: {category} - {len(category_df)} bugs") # Reduce noise, will be logged later
        
    return grouped

async def generate_summary_with_gpt(category, reviews_df, semaphore, max_reviews=25): # Added semaphore, made async
    """Generate summaries, key findings, suggested actions, and priority levels using GPT-3.5 (async)"""
    
    async with semaphore: # Acquire semaphore
        logger.info(f"Acquired semaphore for {category}. Generating summary...")
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
                # Updated code for OpenAI API v1.0.0+ (async)
                completion = await client.chat.completions.create( # await client call
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a software development expert providing bug analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,  # More deterministic
                    max_tokens=1000
                )
                
                response_content = completion.choices[0].message.content
                logger.debug(f"GPT response for {category}: {response_content[:100]}...")
                parsed_response = parse_gpt_response(response_content)

                # Prepare the full result dictionary to be returned
                # This helps in simplifying the main loop later
                return {
                    "bug_category": category,
                    "count": len(reviews_df), # Use original reviews_df for count
                    "summary": parsed_response["summary"],
                    "key_findings": parsed_response["key_findings"],
                    "suggested_actions": parsed_response["suggested_actions"],
                    "priority_level": parsed_response["priority_level"],
                    "additional_notes": parsed_response["additional_notes"],
                    "latest_date": reviews_df['review_date'].max(),
                    "versions_affected": ", ".join(reviews_df['appVersion'].dropna().unique().astype(str))
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API call for {category} failed, retrying in {retry_delay} seconds... ({str(e)})")
                    await asyncio.sleep(retry_delay) # await asyncio.sleep
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Final attempt failed for {category}: {str(e)}")
                    raise # Re-raise the exception to be caught by asyncio.gather
                    
    # This part should ideally not be reached if retries fail and raise an exception.
    # If it is reached due to an unexpected flow, return an error structure.
    # However, asyncio.gather with return_exceptions=True will handle the raised exception.
    # So, this specific structure might not be directly returned if an exception is properly raised.
    # It's more of a fallback if the raise somehow doesn't propagate as expected.
    # For robustness, we'll rely on the exception being caught by gather.
    # If an exception is raised, asyncio.gather will return it for that task.

    # Fallback if all retries fail and an exception wasn't raised (should not happen with current logic)
    logger.error(f"Error generating summary for {category} after all retries.")
    # This return is problematic because an exception should have been raised.
    # Let's ensure an exception is always raised on final failure.
    # The current logic *does* raise, so asyncio.gather will get the exception object.
    # This function will either return the successful dict or an exception will be propagated.
    return None # Should not be reached

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
        if isinstance(result[key], str): # Ensure it's a string before replacing
            # Remove markdown headers like "### 1." or "### 2." 
            result[key] = result[key].replace("### 1.", "").replace("### 2.", "")
            result[key] = result[key].replace("### 3.", "").replace("### 4.", "")
            # Remove asterisks formatting
            result[key] = result[key].replace("****", "")
    
    return result

async def main(): # Make main async
    logger.info("Starting developer summary generation")
    
    bugs_df = load_bug_data()
    grouped_bugs = group_bugs_by_category(bugs_df)
    
    cache = load_cache()
    
    tasks = []
    results_from_cache_or_to_be_processed = [] # Holds data from cache or identifies categories for API calls
    
    semaphore = asyncio.Semaphore(5) # Limit to 5 concurrent OpenAI requests

    categories_to_process_api = {} # Store category_df for API calls

    for category, category_df in grouped_bugs.items():
        if category in cache and is_cache_valid(cache.get(category), category_df):
            logger.info(f"Using cached summary for {category}")
            # The cached data should be the full result structure
            results_from_cache_or_to_be_processed.append(cache[category]['data'])
        else:
            logger.info(f"Cache miss or invalid for {category}. Preparing API call.")
            # Mark for API processing, store category_df
            categories_to_process_api[category] = category_df
            # We will create tasks only for these categories
            
    # Create tasks only for categories not found in cache or with invalid cache
    for category, category_df in categories_to_process_api.items():
        tasks.append(generate_summary_with_gpt(category, category_df, semaphore))

    if tasks:
        logger.info(f"Launching {len(tasks)} API calls concurrently for categories: {list(categories_to_process_api.keys())}...")
        api_call_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results from API calls
        for i, api_result in enumerate(api_call_results_list):
            # Determine category for this result. Order is preserved by gather.
            # This requires tasks list to be in a defined order matching categories_to_process_api.keys()
            category_name_for_this_result = list(categories_to_process_api.keys())[i]
            current_category_df = categories_to_process_api[category_name_for_this_result]

            if isinstance(api_result, Exception):
                logger.error(f"API call for category '{category_name_for_this_result}' failed: {api_result}")
                # Append an error placeholder
                error_data = {
                    "bug_category": category_name_for_this_result,
                    "count": len(current_category_df),
                    "summary": "Error: Generation failed",
                    "key_findings": "Error", "suggested_actions": "Error", "priority_level": "Unknown",
                    "additional_notes": str(api_result),
                    "latest_date": current_category_df['review_date'].max(),
                    "versions_affected": ", ".join(current_category_df['appVersion'].dropna().unique().astype(str))
                }
                results_from_cache_or_to_be_processed.append(error_data)
            elif api_result: # Successful API call, api_result is the full dict from generate_summary_with_gpt
                results_from_cache_or_to_be_processed.append(api_result)
                # Update cache
                cache[api_result['bug_category']] = {
                    "review_count_hash": generate_review_hash(current_category_df), # Use the original df for hash
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": api_result 
                }
    else:
        logger.info("No new API calls needed. All summaries loaded from cache or no categories to process.")

    save_cache(cache)
    
    # Create DataFrame from combined results (cache + new API calls)
    if not results_from_cache_or_to_be_processed:
        logger.warning("No results to process. Output file will be empty or not generated.")
        results_df = pd.DataFrame()
    else:
        results_df = pd.DataFrame(results_from_cache_or_to_be_processed)
    
    # Sort by priority level
    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
        "Unknown": 4
    }
    
    results_df['priority_sort'] = results_df['priority_level'].map(priority_order)
    if not results_df.empty:
        results_df['priority_sort'] = results_df['priority_level'].map(priority_order)
        results_df.sort_values('priority_sort', inplace=True)
        results_df.drop('priority_sort', axis=1, inplace=True)
    
    # Save to CSV
    output_file = "developer_bug_summaries.csv"
    if not results_df.empty:
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
    else:
        logger.info("No data to save. Output file not created.")
        print("\nNo bug summaries were generated or loaded from cache.")


if __name__ == "__main__":
    asyncio.run(main()) # Run the async main