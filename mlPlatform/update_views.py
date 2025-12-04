import re

# Read the file
with open(r'c:\projet final\atelier\mlPlatform\algoML\views.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the Historique.objects.create block
old_pattern = r'(Historique\.objects\.create\(\s+user=request\.user,\s+algo_name=config\[\'name\'\],\s+input_data=donnees,\s+resultat=res_str\s+# On utilise la variable créée plus haut\s+\))'

new_text = '''Historique.objects.create(
                    user=request.user,
                    algo_name=config['name'],
                    input_data=donnees,
                    resultat=res_str,  # On utilise la variable créée plus haut
                    confidence=context.get('confiance', None)  # Score de confiance (None pour régression)
                )'''

content = re.sub(old_pattern, new_text, content, flags=re.MULTILINE | re.DOTALL)

# Write back
with open(r'c:\projet final\atelier\mlPlatform\algoML\views.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("views.py updated successfully!")
