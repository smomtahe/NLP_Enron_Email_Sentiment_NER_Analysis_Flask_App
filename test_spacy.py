import spacy

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Test text with known entities
test_text = """
Message-ID: <18599553.1075842495069.JavaMail.evans@thyme>
Date: Thu, 10 Feb 2000 03:29:00 -0800 (PST)
From: drew.fossum@enron.com
To: bill.cordes@enron.com, dave.neubauer@enron.com, steven.harris@enron.com
Subject: LRC Joint Venture
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Drew Fossum
X-To: Bill Cordes, Dave Neubauer, Steven Harris
X-cc: 
X-bcc: 
X-Folder: \\Drew_Fossum_Dec2000_June2001_1\\Notes Folders\\Sent
X-Origin: FOSSUM-D
X-FileName: dfossum.nsf

I don't see any problem with this transaction since it appears to be limited 
to Louisiana assets, but the issue of whether we are impacted by the 
noncompete agreement strikes me as a commercial call. Please let me know if 
you have any problem with the transaction and I will pursue it. Thanks. DF 
------------------- --- Forwarded by Drew Fossum/ET&S/Enron on 02/10/2000 
11:25 AM --------------------- ------
"""

doc = nlp(test_text)
print("Entities in the text:")
for ent in doc.ents:
    print(ent.text, ent.label_)
