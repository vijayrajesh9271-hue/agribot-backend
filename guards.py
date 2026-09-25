
import re

# list of blocked words
BLOCKED_WORDS = [
     "stupid", "idiot", "hate", "kill"
]

def check_for_abuse(message: str) -> bool:
    """Return True if message contains abusive/offensive content."""
    text = message.lower()
    return any(re.search(rf"\b{word}\b", text) for word in BLOCKED_WORDS)

def handle_abuse_response() -> str:
    """Return a safe response when abuse is detected."""
    return (
        "⚠️ I noticed inappropriate or offensive language in your message. "
        "Let’s keep this conversation respectful so I can help you better."
    )
