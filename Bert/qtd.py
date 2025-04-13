import json

def count_questions(data):
    count = 0
    if isinstance(data, dict):
        if "question" in data:
            count += 1
        for value in data.values():
            count += count_questions(value)
    elif isinstance(data, list):
        for item in data:
            count += count_questions(item)
    return count

# Ajuste o caminho se necessário.
with open(r"dataset.json", encoding="utf-8") as f:
    json_data = json.load(f)

total_questions = count_questions(json_data)
print("Total de perguntas:", total_questions)