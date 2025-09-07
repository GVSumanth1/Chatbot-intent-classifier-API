# intent_mapping.py
# Updated rule-based intent assignment with contextual logic (final version)

import re
import pandas as pd

# ---- INTENT KEYWORDS DICTIONARY (Updated) ----
intent_keywords = {
    "PRODUCT_SEARCH": [
        "buy", "purchase", "order", "looking for", "find", "shop", "browse", "search",
        "get me", "need to buy", "want to order", "I am shopping for", "I want to buy"
    ],
    "VIEW_PRODUCT_DETAILS": [
        "detail", "feature", "specification", "info", "information", "specs", "dimensions",
        "product description", "technical info", "more about this", "full description",
        "material", "breakdown", "in-depth details", "weight", "size", "how big", "how small",
        "product data", "what is it made of", "tell me more about"
    ],
    "CHECK_SALE_ITEMS": [
        "sale", "deal", "discount", "promotion", "offer", "offers", "any discounts today",
        "running promotion", "seasonal sale", "weekly deal", "flash sale", "student discount",
        "today's offer", "coupon available", "redeem offer", "ongoing sale", "any bargains",
        "discounted items"
    ],
    "ADD_TO_CART": [
        "add to cart", "add this", "put in cart", "add item", "include this", "add it to my cart",
        "put this in the bag"
    ],
    "VIEW_CART": [
        "my cart", "shopping cart", "what's in cart", "view cart", "see my cart", "show my cart",
        "what's in my shopping cart", "cart contents"
    ],
    "REMOVE_FROM_CART": [
        "remove from cart", "delete from cart", "take out", "remove item", "discard this",
        "delete this item", "don’t want this"
    ],
    "CLEAR_CART": [
        "empty cart", "clear cart", "remove all", "clear all", "delete everything from cart",
        "clear my shopping bag"
    ],
    "TRACK_ORDER": [
        "track order", "shipment", "shipping status", "where is my order", "delivered",
        "not delivered", "late delivery", "track my item", "order status", "delivery update"
    ],
    "MULTI_ORDER_TRACK": [
        "track my orders", "multiple orders", "all orders", "past orders", "status of all orders"
    ],
    "CANCEL_ORDER": [
        "cancel order", "abort order", "stop order", "don't want it anymore", "cancel my purchase"
    ],
    "RETURN_ITEM": [
        "return", "refund", "exchange", "replace", "send back", "I want to return",
        "request refund", "item return"
    ],
    "VIEW_RETURNS": [
        "returned", "returns history", "my returns", "past returns", "view my returns"
    ],
    "CONNECT_TO_AGENT": [
        "support", "talk to human", "customer care", "help center", "agent", "representative",
        "contact support", "talk to someone", "live agent", "real person", "speak with support",
        "chat with human", "I need help", "connect me to someone", "get assistance",
        "speak to customer care", "Talk to a human", "Connect with a representative",
        "Live support", "Customer service agent", "Human assistance"
    ],
    "FAQ_RETURN_POLICY": [
        "return policy", "how to return", "returns allowed", "can I return", "return instructions",
        "return process", "how do returns work"
    ],
    "FAQ_SHIPPING_TIME": [
        "shipping time", "how long delivery", "delivery time", "when will it arrive",
        "expected delivery", "ETA", "how many days shipping"
    ],
    "FAQ_COD": [
        "cash on delivery", "cod available", "pay on delivery", "cod option", "is cod available",
        "can I pay later"
    ],
    "FAQ_PAYMENT_METHODS": [
        "payment option", "credit card", "debit card", "upi", "paypal", "apple pay", "how to pay",
        "payment methods", "do you accept"
    ],
    "PRODUCT_AVAILABILITY": [
        "in stock", "out of stock", "available", "currently available", "is this available",
        "can I buy this now", "check availability"
    ],
    "VIEW_RECOMMENDATIONS": [
        "recommend", "suggest for me", "similar products", "you may like",
        "what do you suggest", "show me recommendations"
    ],
    "PRODUCT_EXPLAINABILITY": [
        "why recommend", "explain recommendation", "why this", "based on what",
        "why was this shown", "explain why I got this"
    ],
    "SENTIMENT_PRAISE": [
        "amazing", "fantastic", "excellent", "outstanding", "love it", "highly recommend",
        "best purchase", "would buy again", "great product", "superb", "really good",
        "super happy", "totally satisfied"
    ],
    "SENTIMENT_COMPLAINT": [
        "broken", "worst", "never again", "not working", "pathetic", "useless", "hate", "regret",
        "terrible", "awful", "damaged", "late", "disappointed", "poor quality",
        "doesn’t work", "bad experience"
    ],
    "GENERIC_GREETING": [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
        "hello there", "hi there", "good evening", "hey there"
    ],
    "GOODBYE": [
        "bye", "goodbye", "see you", "thank you", "thanks", "see ya", "later",
        "talk soon", "bye bye", "take care","see you later"
    ],
    "OTHER": [
        "blablabla", "xyz product feedback", "what is this even", "not related to purchase",
        "your interface is weird", "I have a random question", "I want to report something else",
        "your name is funny", "do you like ice cream", "what's 2 + 2", "how's the weather",
        "nothing", "random message", "nonsense input", "banana", "i was just typing",
        "no idea", "whatever", "this is not helpful", "makes no sense"
    ]
}


# ---- INTENT ALIAS MAP ----
intent_aliases = {
    "GENERIC_GREETING": "GREETINGS",
    # Add more if needed in future
}
# ---- RULE-BASED INTENT MAPPING ----
def is_greeting(text, greeting_keywords):
    text_lc = text.lower().strip()
    return any(kw in text_lc for kw in greeting_keywords)

def map_intent_conservative_contextual(
    text,
    intent_keywords=intent_keywords,
    default_intent="OTHER",
    ambiguous_intent="AMBIGUOUS"
):
    if not isinstance(text, str):
        return default_intent

    text_lc = text.lower().strip()
    matches = []

    # Match greetings if detected
    if is_greeting(text, intent_keywords.get("GENERIC_GREETING", [])):
        matches.append("GENERIC_GREETING")

    for intent, keywords in intent_keywords.items():
        if intent == "GENERIC_GREETING":
            continue
        for kw in keywords:
            if kw in text_lc:
                matches.append(intent)
                break

    # Apply alias mapping to normalize intent labels
    resolved_matches = [intent_aliases.get(m, m) for m in matches]
    unique_matches = set(resolved_matches)

    if len(unique_matches) == 1:
        return unique_matches.pop()
    elif len(unique_matches) > 1:
        return ambiguous_intent
    else:
        return default_intent
    
    # Rule based intent mapping
def assign_intent_three_tier(row, intent_keywords=intent_keywords):
    rtext = row.get('reviewText', '')
    summ = row.get('summary', '')
    merged = f"{rtext}. {summ}".strip(" .")

    intent_review = map_intent_conservative_contextual(rtext, intent_keywords)
    intent_summary = map_intent_conservative_contextual(summ, intent_keywords)

    if (intent_review == intent_summary) and (intent_review not in ["OTHER", "AMBIGUOUS"]):
        return intent_review, "tier1_both"
    if (intent_review not in ["OTHER", "AMBIGUOUS"]) and (intent_summary in ["OTHER", "AMBIGUOUS"]):
        return intent_review, "tier1_review"
    if (intent_summary not in ["OTHER", "AMBIGUOUS"]) and (intent_review in ["OTHER", "AMBIGUOUS"]):
        return intent_summary, "tier1_summary"
    if (intent_review not in ["OTHER", "AMBIGUOUS"]) and (intent_summary not in ["OTHER", "AMBIGUOUS"]) and (intent_review != intent_summary):
        return "AMBIGUOUS", "tier1_conflict"
    intent_merged = map_intent_conservative_contextual(merged, intent_keywords)
    if intent_merged not in ["OTHER", "AMBIGUOUS"]:
        return intent_merged, "tier2_merged"
    return "AMBIGUOUS", "tier3_model"

def apply_intent_mapping(df, intent_keywords=intent_keywords):
    df[['intent_label', 'confidence_tag']] = df.apply(
        lambda row: pd.Series(assign_intent_three_tier(row, intent_keywords)),
        axis=1
    )
    return df

def inspect_potential_fp(df, intent, tag="tier1_review", n=10):
    subset = df[(df['confidence_tag'] == tag) & (df['intent_label'] == intent)]
    print(f"\n--- Possible false positives for {intent} in {tag} ---")
    if subset.shape[0] == 0:
        print("No examples found.")
        return
    display_cols = ['reviewText', 'summary', 'intent_label', 'confidence_tag']
    print(subset[display_cols].sample(min(n, subset.shape[0]), random_state=123).to_string(index=False))
