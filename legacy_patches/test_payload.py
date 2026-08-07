import json
with open('/services/model-with-adpter-logic-semantic/logs/emergent_2kb_matrix.jsonl', 'r') as f:
    d = json.loads(f.read())
print('Keys:')
for k, v in d.items():
    if k == 'metadata':
        for mk, mv in v.items():
            print(f' - {mk}: {len(str(mv))} bytes')
    else:
        print(f' - {k}: {len(str(v))} bytes')
