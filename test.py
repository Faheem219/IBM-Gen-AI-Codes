def detect_intent(text):
    if "order" or "buy" in text:
        return "order"
    return "UNKNOWN"

print(detect_intent("I want to a pizza"))