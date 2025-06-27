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


"""
==== NLU Output ====
Intent: BookFlight
Entities: [('Delhi', 'GPE'), ('next Tuesday', 'DATE')]

==== NLG Output ====
The user wants to bookflight to Delhi on next Tuesday. Generate a reply.
What happens when the user has asked for a reservation?
In case the user is a user who is not familiar with the internet or the service(s) offered by the airline or other carrier, he will probably ask for a reservation on the next Tuesday. It is best to read the information from the airline or the service(s) which are offered by the airline or other carrier that is in the passenger's travel history and to check for the most suitable reservation under the heading of the reservation.
What happens if the user has asked for a reservation only after checking that the reservation was made in the last 10 days (for example, last 12 months or last 12 months)?   
If the user has asked for a reservation for a long time, he will probably ask for a reservation for the last 10 days. If the user has asked for a reservation for more than one day, he will probably ask for a reservation for the last 10 days.
What happens if the user has asked for a reservation for the last 15 days (for example, last 15 months)?
If the user has asked for a reservation for the last 15 days, he will likely have to change his travel plans.

"""
