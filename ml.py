import pandas as pd
import time
import pickle
import os
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. CHECK LIBRARIES
try:
    from duckduckgo_search import DDGS
    import google.generativeai as genai
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
except ImportError as e:
    print(f"❌ CRITICAL LIBRARY ERROR: {e}")
    print("👉 Run this command: py -m pip install -U google-generativeai duckduckgo-search scikit-learn pandas")
    sys.exit()

# ==========================================
# CONFIGURATION
# ==========================================
GEMINI_API_KEY = "AIzaSyDJeqcDoM3qK82q14s1MtwSxpmM7ffFXR0" 
INPUT_FILE = "input.txt"
DATASET_FILE = "training_data.csv"
MODEL_FILE = "my_model.pkl"

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# PART 0: AUTO-DETECT MODEL
# ==========================================
def get_best_model():
    """Automatically finds a working model name."""
    print("🔌 Connecting to Google AI to find working models...")
    try:
        # List all models that support content generation
        model_list = genai.list_models()
        available_models = [m.name for m in model_list if 'generateContent' in m.supported_generation_methods]
        
        # Prioritize 1.5-flash (fastest), then pro, then any gemini
        preferred_order = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.0-pro']
        
        selected_model = None
        for pref in preferred_order:
            if pref in available_models:
                selected_model = pref
                break
        
        # If no preferred match, take the first available 'gemini' model
        if not selected_model:
            for m in available_models:
                if 'gemini' in m:
                    selected_model = m
                    break
        
        if selected_model:
            print(f"   ✅ Auto-selected model: {selected_model}")
            return genai.GenerativeModel(selected_model)
        else:
            print("   ❌ No Gemini models found available for this API key.")
            print(f"   Available models were: {available_models}")
            return None

    except Exception as e:
        print(f"   ❌ Error connecting to Google AI: {e}")
        return None

# Initialize Model
model_gemini = get_best_model()

# ==========================================
# PART 1: THE RESEARCHER
# ==========================================
def get_internet_description(query_name):
    print(f"🔍 Searching: {query_name}...")
    try:
        with DDGS() as ddgs:
            # Using 'duckduckgo_search' safely
            results = list(ddgs.text(f"{query_name} youtube channel wikipedia", max_results=1))
            if results:
                return results[0]['body']
    except Exception as e:
        print(f"   ⚠️ Search Error: {e}")
    return None

# ==========================================
# PART 2: THE JUDGE
# ==========================================
def get_label_from_ai(description,h):
    if not description or not model_gemini:
        return None 
    
    prompt = f"""
    Classify this text into EXACTLY ONE category: 
    [Tech, Gaming, Education, Vlog, Food, Fitness, Comedy, Motivation, Business, Lifestyle].
    
    Text: "{description}" handle name:"{h}"
    
    Return ONLY the category name.
    """
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"   ⚠️ AI Labeling Failed: {e}")
        return None

# ==========================================
# PART 3: THE BUILDER
# ==========================================
def build_dataset():
    if not model_gemini:
        print("❌ Cannot build dataset: AI Model failed to load.")
        return

    # Delete old bad data
    if os.path.exists(DATASET_FILE):
        os.remove(DATASET_FILE)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Create '{INPUT_FILE}' first.")
        return

    with open(INPUT_FILE, 'r') as f:
        handles = [line.strip() for line in f.readlines() if line.strip()]

    data = []
    print(f"🚀 Starting processing for {len(handles)} handles...")

    for handle in handles:
        desc = get_internet_description(handle)
        
        if desc:
            label = get_label_from_ai(desc,handle)
            if label:
                print(f"   ✅ {handle} -> {label}")
                data.append({"Handle": handle, "Description": desc, "Label": label})
            else:
                print(f"   ❌ Failed to label {handle}")
        else:
            print(f"   ⚠️ No description found for {handle}")
        
        time.sleep(2) # Safety delay

    if data:
        df = pd.DataFrame(data)
        df.to_csv(DATASET_FILE, index=False)
        print(f"\n🎉 Saved {len(df)} rows to {DATASET_FILE}.")
    else:
        print("\n❌ No data collected.")

# ==========================================
# PART 4: THE TEACHER
# ==========================================
def train_model():
    if not os.path.exists(DATASET_FILE):
        print("❌ No data file found.")
        return

    df = pd.read_csv(DATASET_FILE)
    if len(df) < 2:
        print("❌ Not enough data to train (Need 2+ rows).")
        return

    print(f"🧠 Training on {len(df)} examples...")
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(df['Description'], df['Label'])

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved!")

# ==========================================
# PART 5: THE PREDICTOR
# ==========================================
def predict_live():
    if not os.path.exists(MODEL_FILE):
        print("❌ Train the model first (Option 1).")
        return
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    print("\n--- 🤖 LIVE PREDICTION MODE ---")
    while True:
        q = input("\nEnter Handle (or 'q'): ")
        if q.lower() == 'q': break
        
        desc = get_internet_description(q)
        if desc:
            pred = model.predict([desc])[0]
            print(f"   🎯 Prediction: {pred}")
        else:
            print("   ❌ Not found.")

if __name__ == "__main__":
    choice = input("1: Build/Train\n2: Predict\nChoose: ")
    if choice == "1":
        build_dataset()
        train_model()
    elif choice == "2":
        predict_live()