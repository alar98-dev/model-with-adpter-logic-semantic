with open('/home/alifer/swa-aeon/test_mergulho.py', 'r') as f:
    c = f.read()
c = c.replace(
'''        try:
            parts = alvo.split(':')
            reg = swa.synaptic_address.SynapticRegion(parts[2])
            chan = swa.synaptic_address.SynapticChannel(parts[3])
            addr_obj = SynapticAddress(region=reg, channel=chan, node_id=parts[4])
            
            estado = router.get_node_state(addr_obj)''',
'''        try:
            import json
            raw = router.r.get(alvo.replace(':stream', ':state'))
            estado = json.loads(raw) if raw else None'''
)
with open('/home/alifer/swa-aeon/test_mergulho.py', 'w') as f:
    f.write(c)
