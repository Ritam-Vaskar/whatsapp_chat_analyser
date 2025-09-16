import re
import pandas as pd
from datetime import datetime

def parse_whatsapp_chat_from_upload(uploaded_file):
    """
    Parse uploaded WhatsApp chat and return a clean DataFrame.
    """
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()

    # e.g. 18/08/2025, 10:38 pm - Alice: Hello!
    pattern = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM))\s-\s([^:]+):\s(.+)',
        re.IGNORECASE
    )

    messages = []
    for line in lines:
        m = pattern.match(line.strip())
        if not m:
            continue

        date, time, sender, message = m.groups()
        sender = sender.strip()

        # skip blank/very short text
        if not sender or len(message.strip()) < 1:
            continue

        # accept 2-digit or 4-digit year
        for fmt in ("%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p"):
            try:
                ts = datetime.strptime(f"{date} {time}", fmt)
                messages.append({
                    "timestamp": ts,
                    "sender": sender,
                    "message": message.strip()
                })
                break
            except ValueError:
                continue

    return pd.DataFrame(messages)
