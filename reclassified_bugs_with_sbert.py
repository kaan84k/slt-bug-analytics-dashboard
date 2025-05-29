# === Import Libraries ===
# === Import Libraries ===
import pandas as pd
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import torch
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Determine Device for SentenceTransformer ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device} for SentenceTransformer models")

# === Load Dataset ===
file_path = 'categorized_bugs.csv'  # Upload your file here
df = pd.read_csv(file_path)

# === Clean Text ===
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['cleaned_text'] = df['review_description'].apply(preprocess)

# === Split into Labeled and 'Other' ===
labeled_df = df[df['bug_category'].str.lower() != 'other'].copy()
other_df = df[df['bug_category'].str.lower() == 'other'].copy()

# === Load Pretrained SentenceTransformer ===
# Model will be loaded onto the specified device
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Fast & accurate

# === Encode Descriptions ===
# Batch size for encoding. If using GPU, can be increased depending on GPU memory (e.g., 32, 64, 128).
# The model.encode() method will use the device the model is loaded on.
batch_size_encode = 32 # Default batch size, can be tuned.
labeled_embeddings = model.encode(labeled_df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=batch_size_encode)
other_embeddings = model.encode(other_df['cleaned_text'].tolist(), show_progress_bar=True, batch_size=batch_size_encode)

# === Train Classifier ===
classifier = LogisticRegression(max_iter=1000)
classifier.fit(labeled_embeddings, labeled_df['bug_category'])

# === Predict for "Other" ===
predictions = classifier.predict(other_embeddings)
other_df['predicted_category'] = predictions

# === Optional: Cluster for Emerging Categories ===
# Apply PCA before K-Means if there are embeddings
if other_embeddings.shape[0] > 0 and other_embeddings.shape[1] > 0: # Ensure embeddings are not empty
    # Apply PCA to reduce dimensionality for K-Means
    # n_components_pca can be a fixed number (e.g., 50-100) or chosen to explain a certain variance (e.g., 0.95)
    # For simplicity, let's start with a fixed number. Optimal values might require experimentation.
    # If n_samples < n_features, PCA n_components should be <= n_samples.
    n_samples = other_embeddings.shape[0]
    n_features = other_embeddings.shape[1]
    
    # Choose n_components carefully: min(n_samples, n_features, desired_components)
    # Let's aim for 50 components (tunable), but not more than available samples or features.
    desired_components = 50 
    n_components_pca = min(n_samples, n_features, desired_components)

    if n_components_pca > 1: # PCA needs at least 2 components typically, and more than 0 samples
        logger.info(f"Applying PCA to other_embeddings, reducing dimensions to {n_components_pca}")
        pca = PCA(n_components=n_components_pca, random_state=42)
        other_embeddings_pca = pca.fit_transform(other_embeddings)
        logger.info(f"Shape of embeddings after PCA: {other_embeddings_pca.shape}")
    else:
        logger.warning(f"Not enough samples ({n_samples}) or features ({n_features}) to apply PCA effectively with desired_components={desired_components}. Using original embeddings for K-Means if possible.")
        other_embeddings_pca = other_embeddings # Use original embeddings if PCA is skipped
else:
    logger.warning("No 'Other' embeddings to process for K-Means or embeddings are empty. Skipping K-Means and PCA.")
    other_embeddings_pca = other_embeddings # Assign to avoid error later if it's empty (it will be an empty array if other_embeddings is empty)

n_clusters = 5  # You can tune this

# Use PCA-transformed data for K-Means
if other_embeddings_pca.shape[0] > 0 and other_embeddings_pca.shape[0] >= n_clusters: # Check if there are samples to cluster and enough samples for n_clusters
    logger.info(f"Running K-Means clustering with {n_clusters} clusters on PCA-reduced data (shape: {other_embeddings_pca.shape}).")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # Added n_init='auto'
    # Use .loc with original index to ensure correct assignment even if other_df is a slice or has a non-standard index
    other_df.loc[other_df.index, 'cluster'] = kmeans.fit_predict(other_embeddings_pca)
elif other_embeddings_pca.shape[0] > 0 : # Not enough samples for specified n_clusters
     logger.warning(f"Not enough samples ({other_embeddings_pca.shape[0]}) for K-Means with {n_clusters} clusters. Assigning all to cluster 0.")
     other_df.loc[other_df.index, 'cluster'] = 0 # Assign all to cluster 0
else:
    logger.warning("Skipping K-Means clustering as there are no embeddings to cluster (PCA output shape: {other_embeddings_pca.shape}).")
    # Ensure 'cluster' column exists if subsequent code expects it, fill with a default
    other_df.loc[other_df.index, 'cluster'] = -1 # Using -1 to denote no cluster or skipped clustering

# === Visualize Clusters (Optional) ===


# === Merge Updated Data ===
# Replace old 'other' category with predictions
other_df['bug_category'] = other_df['predicted_category']
final_df = pd.concat([labeled_df, other_df], ignore_index=True)


# === Final Keyword-Based Categorization Pass ===
# Using the comprehensive keywords from bug__categories_v2.py (which should be the same as in updated bug__categories.py)
bug_keywords_final_pass = {
    "Login Error": ["login", "log in", "sign in", "cannot access", "authentication", "credentials", "password", "username", "account", "invalid", "oops something went wrong", "verification", "10 digits", "can't log in"],
    "UI Issue": ["layout", "screen", "button", "display", "alignment", "responsive", "interface"],
    "Crash/Freeze": ["crash", "hang", "freeze", "stuck", "unresponsive", "not responding"],
    "Payment Issue": ["payment", "bill", "topup", "recharge", "transaction", "fail", "credit card"],
    "Slow Performance": ["slow", "lag", "delay", "loading", "takes time", "wait", "performance"],
    "Notification Problem": ["notification", "alert", "reminder", "not getting", "missing notifications"],
    "Update Issue": ["update", "updated", "version", "after update", "since update", "new version"],
    "Server Error": ["server error", "server down", "server not responding", "cannot connect", "connection error", "500", "503"],
    "Other": [] 
}

# Ensure 'cleaned_text' exists (it should from earlier in the script, but re-check or re-create if needed)
if 'cleaned_text' not in final_df.columns:
    logger.info("Re-creating 'cleaned_text' column for final keyword pass as it was missing.")
    final_df['cleaned_text'] = final_df['review_description'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

logger.info("Applying final keyword-based categorization pass...")

compiled_regexes_final = []
for category, keywords in bug_keywords_final_pass.items():
    if not keywords: # Skip if category has no keywords (e.g. "Other" if it's empty)
        continue
    pattern = r'|'.join([re.escape(kw) for kw in keywords])
    compiled_regexes_final.append((category, re.compile(pattern, flags=re.IGNORECASE)))

# Create a temporary column for this final keyword pass
final_df['final_keyword_category'] = "Other" 

for category, regex_pattern in compiled_regexes_final:
    # Apply this regex to rows that are still "Other" in the 'final_keyword_category' column
    mask = (final_df['final_keyword_category'] == "Other") & \
           final_df['cleaned_text'].str.contains(regex_pattern, na=False, regex=True)
    final_df.loc[mask, 'final_keyword_category'] = category

# Override SBERT's 'bug_category' if 'final_keyword_category' is not "Other"
# This ensures that specific keyword matches take precedence over SBERT's general classification.
override_mask = final_df['final_keyword_category'] != "Other"
final_df.loc[override_mask, 'bug_category'] = final_df.loc[override_mask, 'final_keyword_category']

final_df.drop(columns=['final_keyword_category'], inplace=True)
logger.info("Final keyword-based categorization pass complete. 'bug_category' column updated.")

# === Save Final Output ===
final_df.to_csv('reclassified_bugs_with_sbert.csv', index=False)
logger.info("✅ Reclassified data with final keyword override saved to: reclassified_bugs_with_sbert.csv")
# print("✅ Reclassified data saved to: /content/reclassified_bugs_with_sbert.csv")

