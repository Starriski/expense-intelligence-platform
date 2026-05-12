from dataclasses import dataclass
from datetime import datetime

@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    date: datetime
    merchant: str
    amount: float
    category: str
    payment_mode: str
    location: str =None