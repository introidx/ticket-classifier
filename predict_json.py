from summerize import extract_summary_and_ids
from predict import predict

def process_ticket(ticket_id: str, ticket_text: str) -> dict:
    # Step 1: Predict category
    category = predict(ticket_text)

    # Step 2: Extract summary, purchaseOrderId, trackingNumber
    extracted = extract_summary_and_ids(ticket_text)

    return {
        "ticketId": ticket_id,
        "category": category,
        "purchaseOrderId": extracted.get("purchaseOrderId"),
        "trackingNumber": extracted.get("trackingNumber"),
        "summary": extracted.get("summary")
    }

if __name__ == "__main__":
    ticket_id = "JIRA-2025"
    ticket_text = "Hi, for PO-12345 and tracking number TRK-78910, I was shown $120 before shipment but billed $150 after delivery."

    result = process_ticket(ticket_id, ticket_text)
    print(result)
