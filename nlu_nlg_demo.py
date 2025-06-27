import spacy
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "Book a flight to Delhi next Tuesday."

# Run NLU: extract entities and intent
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]

if "book" in text.lower() and "flight" in text.lower():
    intent = "BookFlight"
else:
    intent = "Unknown"

print("==== NLU Output ====")
print("Intent:", intent)
print("Entities:", entities)

# Run NLG: generate response using GPT-2
generator = pipeline("text-generation", model="gpt2")
prompt = f"The user wants to {intent.lower()} to {entities[0][0]} on {entities[1][0]}. Generate a reply."

response = generator(prompt, max_length=50, num_return_sequences=1)

print("\n==== NLG Output ====")
print(response[0]["generated_text"])
