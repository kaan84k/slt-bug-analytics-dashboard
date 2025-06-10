import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Step 1: Load and preprocess review data
logger.info("Loading review data...")
df = pd.read_csv('data/slt_selfcare_google_reviews.csv')

# Data exploration
logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Columns: {df.columns.tolist()}")
logger.info(f"Missing values: \n{df.isnull().sum()}")

# Preprocess the reviews
def preprocess_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_bug_keywords(text):
    count = 0
    text = text.lower()
    for keyword in bug_keywords:
        if keyword in text:
            count += 1
    return count

# Handle missing values and preprocess
df['review_description'] = df['review_description'].fillna('').apply(preprocess_text)
df['processed_review'] = df['review_description'].apply(preprocess_text)

# Add additional features
df['review_length'] = df['processed_review'].apply(len)
df['word_count'] = df['processed_review'].apply(lambda x: len(x.split()) if x else 0)

# Check for keywords that might indicate bugs
bug_keywords = ['crash', 'bug', 'error', 'freeze', 'frozen', 'stuck', 'fix',
               'issue', 'problem', 'glitch', 'not working', 'broken', 'failed']

def count_bug_keywords(text):
    count = 0
    text = text.lower()
    for keyword in bug_keywords:
        if keyword in text:
            count += 1
    return count

df['bug_keyword_count'] = df['processed_review'].apply(count_bug_keywords)

# Extract ratings as a feature (if available)
if 'rating' in df.columns:
    # Convert to numeric, fill missing with median
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    median_rating = df['rating'].median()
    df['rating'] = df['rating'].fillna(median_rating)

# Step 2: Embed reviews using a pretrained model
logger.info("Loading transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality

# Only encode non-empty reviews
non_empty_indices = df.index[df['processed_review'] != ''].tolist()
non_empty_reviews = df.loc[non_empty_indices, 'processed_review'].tolist()

logger.info(f"Encoding {len(non_empty_reviews)} non-empty reviews...")
batch_size = 32  # Adjust based on your memory constraints
review_embeddings = np.zeros((len(df), model.get_sentence_embedding_dimension()))

# Process in batches with progress bar
for i in tqdm(range(0, len(non_empty_reviews), batch_size)):
    batch = non_empty_reviews[i:i+batch_size]
    batch_embeddings = model.encode(batch)
    for j, idx in enumerate(non_empty_indices[i:i+batch_size]):
        review_embeddings[idx] = batch_embeddings[j]

# Step 3: Create a larger labeled training dataset
# Expanded manually labeled examples with more diverse patterns
labeled_examples = [
    ("App crashes every time I open it", 1),
    ("Keeps freezing after login", 1),
    ("Very helpful app!", 0),
    ("Great design, but needs dark mode", 0),
    ("Bug: Can't upload images", 1),
    ("Everything works smoothly", 0),
    ("Crash on opening camera feature", 1),
    ("I love the app", 0),
    ("The app won't start after the latest update", 1),
    ("Best app I've ever used", 0),
    ("Excellent features and very intuitive", 0),
    ("Error message appears when trying to save", 1),
    ("App is unstable and keeps closing", 1),
    ("Worth every penny, highly recommend", 0),
    ("Stuck on loading screen", 1),
    ("Could use more customization options", 0),
    ("Perfect for my needs", 0),
    ("The UI is beautiful and responsive", 0),
    ("Can't log in, keeps saying invalid credentials", 1),
    ("Videos won't play, just shows black screen", 1)
]

train_texts, train_labels = zip(*labeled_examples)
train_texts_processed = [preprocess_text(text) for text in train_texts]
train_embeddings = model.encode(train_texts_processed)

# Step 4: Train and optimize classifier
logger.info("Training and optimizing classifier...")

# Split data for proper evaluation
X_train, X_test, y_train, y_test = train_test_split(
    train_embeddings, train_labels, test_size=0.25, random_state=42, stratify=train_labels
)

# Hyperparameter optimization
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
logger.info(f"Best parameters: {best_params}")

# Train final model with best parameters
clf = LogisticRegression(
    **best_params,
    max_iter=1000,
    random_state=42
)
clf.fit(train_embeddings, train_labels)

# --- Semi-supervised self-training step ---
logger.info("Starting semi-supervised self-training...")

# Predict on all non-empty reviews using the initial classifier
pseudo_probs = clf.predict_proba(review_embeddings[non_empty_indices])[:, 1]

# Define high-confidence thresholds
high_bug = 0.95
low_bug = 0.05

# Select high-confidence pseudo-labeled samples
pseudo_bug_indices = [idx for idx, prob in zip(non_empty_indices, pseudo_probs) if prob > high_bug]
pseudo_nonbug_indices = [idx for idx, prob in zip(non_empty_indices, pseudo_probs) if prob < low_bug]

logger.info(f"Pseudo-labeled bugs: {len(pseudo_bug_indices)}, non-bugs: {len(pseudo_nonbug_indices)}")

# Prepare new training data
pseudo_texts = df.loc[pseudo_bug_indices + pseudo_nonbug_indices, 'processed_review'].tolist()
pseudo_labels = [1] * len(pseudo_bug_indices) + [0] * len(pseudo_nonbug_indices)

# Only add if there are enough pseudo-labeled samples
if len(pseudo_texts) > 0:
    pseudo_embeddings = model.encode(pseudo_texts)
    # Combine with original labeled data
    all_embeddings = np.vstack([train_embeddings, pseudo_embeddings])
    all_labels = np.concatenate([train_labels, pseudo_labels])
    # Retrain classifier
    clf.fit(all_embeddings, all_labels)
    logger.info(f"Retrained classifier with {len(all_labels)} samples (including pseudo-labeled)")
else:
    logger.info("Not enough high-confidence pseudo-labeled samples to retrain.")
# --- End semi-supervised self-training ---

# Step 5: Evaluate the model
logger.info("Evaluating model performance...")

# Cross-validation scores
cv_scores = cross_val_score(clf, train_embeddings, train_labels, cv=5, scoring='f1')
logger.info(f"Cross-validation F1 scores: {cv_scores}")
logger.info(f"Mean F1 score: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

# Evaluate on test set
y_pred = clf.predict(X_test)
class_report = classification_report(y_test, y_pred)
logger.info(f"\nClassification Report on Test Set:\n{class_report}")

# Step 6: Predict bugs in full dataset
logger.info("Predicting bug reports in the full dataset...")
df['is_bug_report'] = 0  # Default for empty reviews

# Only predict for non-empty reviews
df.loc[non_empty_indices, 'is_bug_report'] = clf.predict(review_embeddings[non_empty_indices])
df.loc[non_empty_indices, 'bug_confidence'] = clf.predict_proba(review_embeddings[non_empty_indices])[:, 1]

# For empty reviews, set confidence to 0
df['bug_confidence'] = df['bug_confidence'].fillna(0)

# Step 7: Enhanced output with additional insights
logger.info("Generating insights and saving results...")

# Summarize findings
bug_count = df['is_bug_report'].sum()
total_reviews = len(df)
bug_percentage = (bug_count / total_reviews) * 100 if total_reviews > 0 else 0

logger.info(f"Total reviews analyzed: {total_reviews}")
logger.info(f"Bug reports identified: {bug_count} ({bug_percentage:.2f}%)")

# Calculate high confidence bugs (e.g., confidence > 0.8)
high_conf_bugs = df[df['bug_confidence'] > 0.8]
logger.info(f"High confidence bug reports: {len(high_conf_bugs)}")

# Sort by confidence for prioritization
bugs_prioritized = df[df['is_bug_report'] == 1].sort_values('bug_confidence', ascending=False)

# Save detailed results
output_columns = [
    'review_id', 'user_name', 'review_description', 'review_date', 'appVersion', 'rating',
    'is_bug_report', 'bug_confidence', 'bug_keyword_count', 'review_length'
]

# Ensure columns exist in the DataFrame before selecting
available_columns = [col for col in output_columns if col in df.columns]

# Save detailed results for all reviews
df[available_columns].to_csv('data/bug_predictions.csv', index=False)

# Save prioritized bugs for easy review
if not bugs_prioritized.empty:
    # Ensure columns exist in the prioritized DataFrame before selecting
    available_prioritized_columns = [col for col in output_columns if col in bugs_prioritized.columns]
    bugs_prioritized[available_prioritized_columns].to_csv('data/prioritized_bugs.csv', index=False)


logger.info("\n✅ Bug detection completed. Results saved to:")
logger.info("  - bug_predictions.csv (all reviews)")
logger.info("  - prioritized_bugs.csv (bug reports sorted by confidence)")