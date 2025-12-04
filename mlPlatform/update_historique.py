import re

# Read the file
with open(r'c:\projet final\atelier\mlPlatform\algoML\templates\historique.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Add Score de Confiance column to header
content = content.replace(
    '''                                <th>Résultat</th>
                            </tr>''',
    '''                                <th>Résultat</th>
                                <th>Score de Confiance</th>
                            </tr>'''
)

# Add confidence score display in table body
old_result_cell = '''                                
                                <td>
                                    <span class="fw-bold text-success">{{ item.resultat }}</span>
                                </td>
                            </tr>'''

new_result_cell = '''                                
                                <td>
                                    <span class="fw-bold text-success">{{ item.resultat }}</span>
                                </td>
                                
                                <td>
                                    {% if item.confidence %}
                                        {% if item.confidence >= 90 %}
                                            <span class="badge bg-success">{{ item.confidence }}%</span>
                                        {% elif item.confidence >= 75 %}
                                            <span class="badge bg-info">{{ item.confidence }}%</span>
                                        {% elif item.confidence >= 60 %}
                                            <span class="badge bg-warning text-dark">{{ item.confidence }}%</span>
                                        {% else %}
                                            <span class="badge bg-danger">{{ item.confidence }}%</span>
                                        {% endif %}
                                    {% else %}
                                        <span class="text-muted">N/A</span>
                                    {% endif %}
                                </td>
                            </tr>'''

content = content.replace(old_result_cell, new_result_cell)

# Write back
with open(r'c:\projet final\atelier\mlPlatform\algoML\templates\historique.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("historique.html updated successfully!")
