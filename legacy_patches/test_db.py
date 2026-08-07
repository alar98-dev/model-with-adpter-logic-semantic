import traceback
from swa.topological_db import registrar_leitura_vetorial, _db_cache, TopologicalDB
import time

try:
    print("Testing DB insert...")
    db = TopologicalDB()
    if db._model is None:
        print("Model is none!")
    else:
        print("Model loaded.")
    
    registrar_leitura_vetorial("/test/path", "This is a test content", 0.5)
    time.sleep(5)  # Wait for thread to finish
    print("Done waiting.")
except Exception as e:
    traceback.print_exc()
