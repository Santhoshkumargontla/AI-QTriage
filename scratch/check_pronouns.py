import docx

doc = docx.Document("AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx")
first_person = []
pronouns = {"we", "our", "us", "my", "i"}

for i, p in enumerate(doc.paragraphs):
    for w in p.text.split():
        clean = w.lower().strip(".,;:()[]\"'")
        if clean in pronouns:
            first_person.append((i, clean, p.text[:120]))
            break

print(f"Total first-person occurrences found: {len(first_person)}")
for idx, w, snippet in first_person:
    print(f"P{idx} [{w}]: {snippet}...")
