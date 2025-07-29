import openai
import os
import json
from dotenv import load_dotenv
import re

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
openai_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
temperature = float(os.environ.get("TEMPERATURE", 0.3))

def build_extraction_prompt(text: str) -> str:
    return (
        "You are an AI assistant that reads customer support tickets and extracts key information.\n\n"
        "From the ticket below, extract:\n"
        "- A one-line summary of the ticket\n"
        "- The purchase order ID (if any). This may appear in the ticket as 'poId', 'purchaseOrderId', 'purchase order id', or similar variations.\n"
        "- The tracking number (if any). This may appear as 'trackingNumber', 'tracking_id', 'tracking number', 'trackingId', or similar variations.\n\n"
        "Respond ONLY in valid JSON format with exactly these keys: summary, purchaseOrderId, trackingNumber.\n"
        "If a value is not found, return it as null (not the string 'null').\n\n"
        f"Ticket:\n{text}\n\n"
        "Your JSON:"
    )

def call_openai_api(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model=openai_model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You extract structured data from customer support messages."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

def extract_summary_and_ids(text: str) -> dict:
    prompt = build_extraction_prompt(text)
    try:
        content = call_openai_api(prompt)
        return safe_parse_json(content)
    except Exception as e:
        print("OpenAI API failed:", e)
        return {
            "summary": None,
            "purchaseOrderId": None,
            "trackingNumber": None
        }

def safe_parse_json(response_text):
    try:
        json_str = re.search(r'{.*}', response_text, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        print("Failed to parse JSON:", e)
        return {
            "summary": None,
            "purchaseOrderId": None,
            "trackingNumber": None
        }    
    
# Example usage
if __name__ == "__main__":
    ticket_text = "Why does the invoice show a higher charge than what was estimated before dispatch with purchaseOrderId 12345 and trackingNumber 67890?"
    result = extract_summary_and_ids(ticket_text)
    print("Extracted Information:", result)    
