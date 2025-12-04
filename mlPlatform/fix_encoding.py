import re

# Read the file
with open(r'c:\projet final\atelier\mlPlatform\algoML\views.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove emojis from print statements to avoid encoding issues
content = content.replace('⚡', '')
content = content.replace('✅', '')
content = content.replace('⚠️', '')
content = content.replace('❌', '')

# Write back
with open(r'c:\projet final\atelier\mlPlatform\algoML\views.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed emojis from views.py")
