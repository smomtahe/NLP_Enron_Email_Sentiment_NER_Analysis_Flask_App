import email # standard library in Python for handling email messages.
from email import policy # defines policies that govern the behavior of the email package
from email.parser import BytesParser # parse email messages from bytes.

def parse_email(raw_email):
    parsed_email = {} # store the extracted components of the email.
    msg = BytesParser(policy=policy.default).parsebytes(raw_email.encode()) 
    
    parsed_email['body'] = msg.get_body(preferencelist=('plain')).get_content()
    parsed_email['subject'] = msg['subject']
    parsed_email['from'] = msg['from']
    parsed_email['to'] = msg['to']
    
    return parsed_email

# Parsing email messages from bytes means taking raw email data, which is often received as a byte stream (a sequence of bytes), and converting it into a structured format that allows for easy access to the email's components, such as the subject, sender, recipient, and body.

