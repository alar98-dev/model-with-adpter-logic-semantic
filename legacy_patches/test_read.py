with open('/services/model-with-adpter-logic-semantic/logs/emergent_2kb_matrix.jsonl', 'rb') as f:
    data = f.read()
    print(len(data))
    print(data[:50])
    print(data[-50:])
