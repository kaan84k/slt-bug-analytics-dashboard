# === Import Libraries ===
import pandas as pd
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & accurate

# === Encode Descriptions ===
labeled_embeddings = model.encode(labeled_df['cleaned_text'].tolist(), show_progress_bar=True)
other_embeddings = model.encode(other_df['cleaned_text'].tolist(), show_progress_bar=True)

# === Train Classifier ===
classifier = LogisticRegression(max_iter=1000)
classifier.fit(labeled_embeddings, labeled_df['bug_category'])

# === Predict for "Other" ===
predictions = classifier.predict(other_embeddings)
other_df['predicted_category'] = predictions

# === Optional: Cluster for Emerging Categories ===
n_clusters = 5  # You can tune this
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
other_df['cluster'] = kmeans.fit_predict(other_embeddings)

# === Visualize Clusters (Optional) ===


# === Merge Updated Data ===
# Replace old 'other' category with predictions
other_df['bug_category'] = other_df['predicted_category']
final_df = pd.concat([labeled_df, other_df], ignore_index=True)

# === Save Final Output ===
final_df.to_csv('reclassified_bugs_with_sbert.csv', index=False)
# print("✅ Reclassified data saved to: /content/reclassified_bugs_with_sbert.csv")

