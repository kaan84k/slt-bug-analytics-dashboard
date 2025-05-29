# -*- coding: utf-8 -*-

from google_play_scraper import reviews_all, Sort
import pandas as pd

def fetch_google_reviews(app_id, lang='en', country='us'):
    try:
        print(f"Fetching reviews for: {app_id}")
        reviews = reviews_all(
            app_id,
            sleep_milliseconds=0, 
            lang=lang,
            country=country,
            sort=Sort.NEWEST
        )
        
        df = pd.DataFrame(reviews)
        
        # Drop unwanted columns
        df.drop(columns=['userImage', 'reviewCreatedVersion'], errors='ignore', inplace=True)
        
        # Rename columns for clarity
        rename_map = {
            'score': 'rating',
            'userName': 'user_name',
            'reviewId': 'review_id',
            'content': 'review_description',
            'at': 'review_date',
            'replyContent': 'developer_response',
            'repliedAt': 'developer_response_date',
            'thumbsUpCount': 'thumbs_up'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Add metadata columns
        df.insert(0, 'source', 'Google Play')
        df.insert(2, 'review_title', None)
        df['language_code'] = lang
        df['country_code'] = country
        
        return df
    except Exception as e:
        print(f"Error fetching reviews: {e}")
        return pd.DataFrame()


# Scrape reviews for the SLT-Mobitel Selfcare App
GOOGLE_APP_ID = "com.slt.selfcare"
reviews_df = fetch_google_reviews(GOOGLE_APP_ID)

# Display some of the reviews
print(reviews_df.head())

# Optionally save to CSV
reviews_df.to_csv("slt_selfcare_google_reviews.csv", index=False)