from sentence_transformers import SentenceTransformer
print("Loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("Encoding...")
try:
    vec = model.encode("Hello world")
    print("Success! Shape:", vec.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
