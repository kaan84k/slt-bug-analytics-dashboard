bug_keywords = {
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
    "Connection Management": [
        "add connection", "new connection", "register connection", "add my connection",
        "cant register", "can't register", "cannot register", "register new",
        "prepaid connection", "pre paid", "reload prepaid", "add prepaid",
        "multiple connections", "other connections", "two connections", "three connections",
        "configure connection", "setup connection", "connection setup", "fiber connection",
        "broadband connection", "4g connection", "lte connection", "adsl"
    ],
    "Signal/Network Quality": [
        "service unavailable", "not available", "unavailable", "no service",
        "poor connection", "bad signal", "no internet", "can't connect",
        "connection issue", "service temporarily unavailable", "temporary unavailable",
        "service error", "not working", "app is not working", "service not available",
        "connection problem", "network error", "network issue", "connectivity",
        "connection speed", "slow speed", "low speed", "speed issues",
        "buffering", "loads slow", "slow loading", "higher user base",
        "network speeds", "slow network", "poor network", "hd video loads",
        "video buffering", "streaming issues", "connection quality"
    ],
    "Package/Plan Issues": [
        "package", "plan", "data balance", "usage", "quota", "remaining data",
        "data usage", "package details", "change package", "upgrade package",
        "youth add on", "free offer", "add on", "data plan", "broadband package",
        "package activation", "package not working", "can't activate package",
        "prepaid broadband", "broadband number", "cannot add broadband",
        "package don't work", "package doesn't work", "data package",
        "package upgrade", "package change", "view package", "find package"
    ],
    "PeoTV Issues": [
        "peo tv", "peotv", "channel", "tv service", "television",
        "streaming", "video quality", "channel list", "program",
        "tv connection", "tv package", "channel not working", "tv issues"
    ],
    "Installation Issues": [
        "install", "installation", "cant install", "can't install", "installation failed",
        "download failed", "app store", "play store", "installation error",
        "download error", "unable to install", "won't install"
    ],
    "Complaint Management": [
        "complaint", "feedback", "report issue", "customer service",
        "bad application", "poor app", "terrible app", "worst app",
        "useless app", "waste of time", "bad service", "poor service",
        "horrible app", "awful app", "pathetic app", "complaint status",
        "raise complaint", "lodge complaint", "submit complaint"
    ],
    "Language/Localization": [
        "language", "translation", "english", "sinhala", "tamil",
        "wrong language", "language setting", "text display", "foreign language",
        "language option", "language preference"
    ],
    "System Faults": [
        "system error", "system issue", "system fault", "system problem",
        "application doesn't work", "app not functioning", "functionality issues",
        "system down", "system unavailable", "system failure",
        "something went wrong", "doesn't work properly", "not working properly",
        "application error", "app error", "system error"
    ],
    "Other": []
}

import pandas as pd
import re

# Load prioritized bug data
df = pd.read_csv("data/reclassified_bugs_with_sbert.csv")

# Preprocess review text
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    return text

df['cleaned_description'] = df['review_description'].apply(preprocess)

def categorize_review(text):
    text = text.lower()
    
    # Check categories in order of priority
    priority_order = [
        "Contact Change Issue",
        "Login Error",
        "Payment Issue",
        "Server Error",
        "Connection Management",
        "Signal/Network Quality",
        "Package/Plan Issues",
        "PeoTV Issues",
        "Installation Issues",
        "Complaint Management",
        "Update Issue",
        "Crash/Freeze",
        "Slow Performance",
        "UI Issue",
        "Notification Problem",
        "Language/Localization",
        "System Faults"
    ]
    
    # Check each category in priority order
    for category in priority_order:
        for keyword in bug_keywords[category]:
            if keyword.lower() in text:
                return category
    
    # Generic error handling
    error_keywords = ["error", "not working", "issue", "problem", "bug", "failed", "failure"]
    if any(keyword in text for keyword in error_keywords):
        # Try to categorize generic errors based on context
        if "data" in text or "usage" in text:
            return "Package/Plan Issues"
        elif "wifi" in text or "internet" in text or "connection" in text:
            return "Signal/Network Quality"
        elif "login" in text or "account" in text:
            return "Login Error"
        elif "app" in text and ("launch" in text or "open" in text):
            return "Crash/Freeze"
    
    return "Other"

df['bug_category'] = df['cleaned_description'].apply(categorize_review)

# Save the categorized data
df[['review_description', 'bug_category', 'review_date', 'appVersion']].to_csv("data/categorized_bugs.csv", index=False)
