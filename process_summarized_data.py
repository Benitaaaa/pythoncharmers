import spacy
import pickle
import pandas as pd
import os
import csv
import requests
import re
from collections import defaultdict
from transformers import pipeline, AutoTokenizer
from fitz import open as fitz_open  # PyMuPDF
import openai
import pycountry
from dotenv import load_dotenv
import os



# Load environment variables from .env file
load_dotenv() 

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face sentiment model
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# OpenAI API Client (Set API Key Securely)
client = openai.OpenAI(api_key=os.getenv("GITHUB_API_KEY"))  

class SummarizedTextAnalyzer:
    def __init__(self):
        """Initialize analyzer for extracting relationships from summarized text."""
        self.nlp = nlp
        self.country_relationships = []
        self.organization_relationships = []
        self.country_mentions = defaultdict(int)
        self.organization_mentions = defaultdict(int)

    def clean_text(self, text):
        """Remove special characters, boilerplate content, and extra whitespace."""
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
        return text.strip()
    
    def standardize_country_name(self, country):
        """Attempt to standardize country names using pycountry."""
        try:
            return pycountry.countries.lookup(country).name
        except LookupError:
            return None  # Ignore non-country entities
    
    def normalize_org_name(org_name):
        """Normalize organization names using DBpedia Spotlight API."""
        url = "https://api.dbpedia-spotlight.org/en/annotate"
        headers = {"Accept": "application/json"}
        params = {"text": org_name, "confidence": 0.5}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if "Resources" in data:
                    return data["Resources"][0]["@URI"].split("/")[-1]  # Extract DBpedia title
        except Exception as e:
            print(f"Error normalizing {org_name}: {e}")

        return org_name  # Return original if not found
    
    def get_sentiment(self, sentence):
        """Analyze sentiment while handling long sentences (Max: 512 tokens)."""
        max_length = 512  # Model token limit
        
        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        
        # If the tokenized input is too long, split into chunks
        if len(tokens) > max_length:
            chunks = [" ".join(tokenizer.convert_tokens_to_string(tokens[i:i + max_length])) 
                      for i in range(0, len(tokens), max_length)]
        else:
            chunks = [sentence]  # Use original if within limit
    
        # Analyze sentiment for each chunk separately
        sentiments = []
        for chunk in chunks:
            try:
                result = sentiment_pipeline(chunk)
                sentiments.append(result[0]['label'])
            except RuntimeError as e:
                print(f"Error processing chunk: {e}")
                return "NEUTRAL"  # Default to NEUTRAL if there's an error
    
        # Aggregate sentiment from all chunks
        positive_count = sentiments.count("POSITIVE")
        negative_count = sentiments.count("NEGATIVE")
    
        if positive_count > negative_count:
            return "POSITIVE"
        elif negative_count > positive_count:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    def summarize_text(self, text):
        """Summarize extracted text using GPT-4o-mini."""
        prompt = f"""Identify the main points in the article provided.
        Find relationships involving Organizations and Countries.
        \n\nArticle:\n{text[:4000]}"""  # Truncate to prevent exceeding token limits

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                store=True,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ GPT Summarization Error: {e}")
            return text  # Return original text as fallback

    def process_pdfs(self, pdf_dir, output_pkl="processed_countries_and_organizations.pkl"):
        """Extract text from PDFs, summarize with GPT, extract entities, and save all data to a pickle file."""
        
        if os.path.exists(output_pkl):
            with open(output_pkl, "rb") as f:
                saved_data = pickle.load(f)
                print(f"Loaded existing data from {output_pkl}.")
        else:
            saved_data = {"summarized_texts": [], "country_relationships": [], "organization_relationships": [],
                          "country_mentions": defaultdict(int), "organization_mentions": defaultdict(int)}

        summarized_texts = saved_data["summarized_texts"]

        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_dir, filename)

                if any(entry["filename"] == filename for entry in summarized_texts):
                    print(f"Skipping {filename}, already summarized.")
                    continue

                with fitz_open(filepath) as doc:
                    text = "\n".join([page.get_text("text") for page in doc])

                    summarized_text = self.summarize_text(text)
                    summarized_texts.append({"filename": filename, "summarized_text": summarized_text})

                    self.extract_entities_and_relationships(summarized_text, source=filename)

        saved_data["summarized_texts"] = summarized_texts
        saved_data["country_relationships"] = self.country_relationships
        saved_data["organization_relationships"] = self.organization_relationships
        saved_data["country_mentions"] = self.country_mentions
        saved_data["organization_mentions"] = self.organization_mentions

        with open(output_pkl, "wb") as f:
            pickle.dump(saved_data, f)

        print(f"Processed PDFs and saved data to {output_pkl}.")

    def extract_entities_and_relationships(self, summarized_text, source):
        """Extract country and organization relationships from summarized text."""
        doc = self.nlp(summarized_text)  # Use summarized text ONLY

        for ent in doc.ents:
            if ent.label_ == "GPE":
                country_name = self.standardize_country_name(ent.text)
                if country_name:
                    self.country_mentions[country_name] += 1
            elif ent.label_ == "ORG":
                self.organization_mentions[ent.text] += 1

        # Extract country relationships with sentiment
        for sent in doc.sents:
            entities_in_sent = [self.standardize_country_name(e.text) for e in sent.ents if e.label_ == "GPE"]
            #entities_in_sent = [e.text for e in sent.ents if e.label_ == "GPE"]
            entities_in_sent = [e for e in entities_in_sent if e]

            if len(entities_in_sent) >= 2:
                sentiment = self.get_sentiment(sent.text)  # Sentiment only for summarized text
                self.country_relationships.append({
                    'source': entities_in_sent[0],
                    'target': entities_in_sent[1],
                    'sentence': sent.text,
                    'sentiment': sentiment, 
                    'source_file': source
                })

        # Extract organization relationships with sentiment
        for sent in doc.sents:
            entities_in_sent = [e.text for e in sent.ents if e.label_ == "ORG"]
            if len(entities_in_sent) >= 2:
                sentiment = self.get_sentiment(sent.text) 
                self.organization_relationships.append({
                    'source': entities_in_sent[0],
                    'target': entities_in_sent[1],
                    'sentence': sent.text,
                    'sentiment': sentiment,  
                    'source_file': source
                })

    def load_data(self, filename="processed_countries_and_organizations.pkl"):
        """Load previously processed data from a pickle file."""
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded data from {filename}.")
            return data
        else:
            print(f"No data found in {filename}. Run processing first!")
            return None
if __name__ == "__main__":
    pdf_dir = "pdfs" 

    analyzer = SummarizedTextAnalyzer()
    
    analyzer.process_pdfs(pdf_dir)

    with open("processed_countries_and_organizations.pkl", "rb") as f:
        data = pickle.load(f)

    print(f"Total Country Relationships: {len(data['country_relationships'])}")
    print(f"Total Organization Relationships: {len(data['organization_relationships'])}")
    print("Data processing complete.")
