bug_keywords = {
    "Login Error": [
        "login", "log in", "sign in", "cannot access", "authentication", "credentials",
        "cannot log", "cannot login", "login problem", "login issue", "log my account",
        "log account", "unable to login", "cant login", "signing in", "invalid password",
        "username invalid"
    ],
    "Contact Change Issue": [
        "contact info", "phone number", "mobile number", "change number", "update contact",
        "contact information", "contact details", "phone details", "cannot change",
        "number change", "update number", "edit contact"
    ],
    "UI Issue": [
        "layout", "screen", "button", "display", "alignment", "responsive", "interface",
        "user interface", "ui", "blank screen", "design"
    ],
    "Crash/Freeze": [
        "crash", "crashed", "crashing", "freeze", "freezing", "stuck", "unresponsive",
        "not responding", "force close", "force stop", "hang", "stuck on"
    ],
    "Payment Issue": [
        "payment", "bill", "topup", "recharge", "transaction", "fail", "credit card",
        "unable to pay", "payment failed", "payment error", "bill pay", "transaction error",
        "session timeout"
    ],
    "Slow Performance": [
        "slow", "lag", "delay", "loading", "takes time", "wait", "performance",
        "slow loading", "lagging", "sluggish"
    ],
    "Notification Problem": [
        "notification", "alert", "reminder", "not getting", "missing notifications",
        "notification error", "no notification"
    ],
    "Update Issue": [
        "update", "updated", "version", "after update", "since update", "new version",
        "upgrade", "latest update", "post update"
    ],
    "Server Error": [
        "server error", "server down", "server not responding", "cannot connect",
        "connection error", "500", "503", "internal server error", "system error",
        "service unavailable"
    ],
    "Network/Connection": [
        "network", "connection", "wifi", "internet", "signal", "offline", "no service",
        "poor connection", "weak signal", "disconnected", "no internet", "network error",
        "connection lost"
    ],
    "OTP Issue": [
        "otp", "one time password", "verification code", "verification", "otp code",
        "didnt get otp", "otp not received"
    ],
    "Contact Change Issue": [
        "contact", "phone number", "mobile number", "change number", "update contact",
        "contact information", "contact details", "phone details", "cannot change",
        "number change", "modify contact", "edit number", "update number",
        "change contact", "edit contact", "contact change", "change details",
        "contact not working", "cant change contact", "can't change contact",
        "unable to change contact", "fix contact", "contact issue"
    ],
    "Login Error": [
        "login", "log in", "sign in", "cannot access", "authentication", "credentials",
        "cannot log", "cannot login", "login problem", "login issue", "log my account", "log account"
        "password", "username", "account", "invalid", "oops something went wrong",
        "verification", "10 digits", "can't log in", "cant log", "wrong password",
        "password incorrect", "account access", "sign up", "signup", "register",
        "registration", "cant register", "can't register"
    ],
    "UI Issue": [
        "layout", "screen", "button", "display", "alignment", "responsive", "interface",
        "poor experience", "bad experience", "features not working", "features won't work",
        "app features", "user interface", "ui problems", "ui issues", "design issues",
        "full of bugs", "many bugs", "bugs did not fix", "bugs not fixed"
    ],
    "Crash/Freeze": [
        "crash", "hang", "freeze", "stuck", "unresponsive", "not responding",
        "buggy app", "bugs everywhere", "lot of bugs", "inconsistencies",
        "app crashes", "keeps crashing", "application crash", "stops working",
        "force close", "app hangs", "app freezes", "app stops"
    ],
    "Payment Issue": [
        "payment", "bill", "topup", "recharge", "transaction", "fail", "credit card",
        "online reload", "reload failed", "payment not working", "can't pay",
        "payment error", "billing issue", "payment problem", "transaction failed",
        "payment gateway", "payment unsuccessful", "reload issue"
    ],
    "Slow Performance": [
        "slow", "lag", "delay", "loading", "takes time", "wait", "performance",
        "sluggish", "buffering", "slow loading", "slow response", "takes forever",
        "not fast", "speed issue", "performance issue", "slow app"
    ],
    "Notification Problem": [
        "notification", "alert", "reminder", "not getting", "missing notifications",
        "push notification", "notification not working", "alerts not working",
        "no notifications", "notification settings"
    ],
    "Update Issue": [
        "update", "updated", "version", "after update", "since update", "new version",
        "cant update", "can't update", "update failed", "update error", "latest version",
        "version issue", "upgrade app", "app update", "forced update"
    ],
    "Server Error": [
        "server error", "server down", "server not responding", "cannot connect",
        "connection error", "500", "503", "internal server", "server maintenance",
        "server timeout", "server issue", "backend error"
    ],
    "Other": []
}

import pandas as pd
import re
from db_utils import save_df

# Load prioritized bug data
df = pd.read_csv("data/prioritized_bugs.csv")

# Preprocess review text
def preprocess(text):
    text = str(text).lower()
    text = text.replace("can't", "cannot")
    text = text.replace("can t", "cannot")
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['cleaned_description'] = df['review_description'].apply(preprocess)

def categorize_review(text):
    text = text.lower()
    # Flatten all keywords with their categories and sort by length (longest first)
    keyword_category_pairs = []
    for category, keywords in bug_keywords.items():
        for keyword in keywords:
            keyword_category_pairs.append((keyword.lower(), category))
    keyword_category_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for keyword, category in keyword_category_pairs:
        # Use word boundaries for all keywords (multi-word and single)
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text):
            return category
    return "Other"

df['bug_category'] = df['cleaned_description'].apply(categorize_review)

df[['review_description', 'bug_category', 'review_date', 'appVersion']].to_csv("data/categorized_bugs.csv", index=False)
save_df(df[['review_description', 'bug_category', 'review_date', 'appVersion']], 'categorized_bugs')

# Debug Review Example
if __name__ == "__main__":

    sample_review = "system error is displayed every time when i want to subscribe or unsubscribe a package can t do anything through the app"
    cleaned = preprocess(sample_review)
    category = categorize_review(cleaned)
    print(f"Sample review: {sample_review}")
    print(f"Predicted category: {category}")
