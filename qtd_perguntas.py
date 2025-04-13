import json

def count_questions(data):
    count = 0
    for item in data:
        if 'qas' in item:
            count += len(item['qas'])
    return count

with open(r"dataset_tunado.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total_questions = count_questions(data)
print(f"Total de perguntas no dataset: {total_questions}")