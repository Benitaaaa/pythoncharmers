import requests
from bs4 import BeautifulSoup
import spacy
import pycountry
import re
import pickle
from collections import defaultdict
from transformers import pipeline, AutoTokenizer
import pandas as pd
import pycountry_convert as pc

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face Transformers Sentiment Model
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Load tokenizer for accurate splitting
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

class CountrySentimentAnalyzer:
    def __init__(self):                           
        """Initialize the text analyzer with NLP and data structures."""
        self.nlp = nlp  # Use loaded SpaCy model
        self.entities = defaultdict(set)
        self.relationships = []
        self.country_mentions = defaultdict(int)  # Track how often a country appears

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

    def extract_entities_and_relationships(self, text, source=None):
        """Extract country entities (GPE) and relationships from text."""
        cleaned_text = self.clean_text(text)
        doc = self.nlp(cleaned_text)
        country_entities = set()

        for ent in doc.ents:
            if ent.label_ == "GPE":
                country_name = self.standardize_country_name(ent.text)
                if country_name:  # Only add valid countries
                    country_entities.add(country_name)
                    self.country_mentions[country_name] += 1  # Track country frequency

        # Store valid countries
        for country in country_entities:
            self.entities['GPE'].add((country, source))

        # Create relationships between countries in the same sentence
        for sent in doc.sents:
            sent_doc = self.nlp(sent.text)
            entities_in_sent = [self.standardize_country_name(e.text) for e in sent_doc.ents if e.label_ == "GPE"]
            entities_in_sent = [e for e in entities_in_sent if e]  # Remove None values

            if len(entities_in_sent) >= 2:
                sentiment = self.get_sentiment(sent.text)  # Get sentiment of the sentence
                for i in range(len(entities_in_sent) - 1):
                    self.relationships.append({
                        'source': entities_in_sent[i],
                        'target': entities_in_sent[i + 1],
                        'sentence': sent.text,
                        'sentiment': sentiment
                    })

    def process_csv(self, csv_file):
        """Process CSV file and analyze extracted text."""
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            text = row.get("Text", "").strip()
            if text:
                self.extract_entities_and_relationships(text, source="CSV Data")

    def save_data(self, filename="processed_data.pkl"):
        """Save extracted relationships and mentions to a file."""
        with open(filename, "wb") as f:
            pickle.dump({"relationships": self.relationships, "country_mentions": self.country_mentions}, f)
        print("Data extracted and saved.")

if __name__ == "__main__":
    analyzer = CountrySentimentAnalyzer()
    csv_file = "news_excerpts_parsed.csv"  # Ensure this file exists in your directory
    analyzer.process_csv(csv_file)
    analyzer.save_data()
    print("Data processing complete. You can now use this data for visualization.")
