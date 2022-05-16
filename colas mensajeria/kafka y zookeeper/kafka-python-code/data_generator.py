import random 
import string 
from datetime import datetime

user_ids = list(range(1, 101))
recipient_ids = list(range(1, 101))

def generate_message(message = None, random_user_id=None) -> dict:
    if random_user_id is None:
        random_user_id = random.choice(user_ids)
    # Copy the recipients array
    recipient_ids_copy = recipient_ids.copy()
    # User can't send message to himself
    recipient_ids_copy.remove(random_user_id)
    random_recipient_id = random.choice(recipient_ids_copy)
    #   Generate a random message
    if message is None:
        message = ''.join(random.choice(string.ascii_letters) for i in range(32))
    return {
            'user_id': random_user_id,
            'recipient_id': random_recipient_id,
            'message': message
        }

def design_message(message, user, date=None, grupo=None):
    if date is None:
        date = datetime.now()
        date = str(date)
    return {
            'user_id': user,
            'date': date,
            'message': message,
            'chat_id':grupo
        }