from modal_jobs.common import HF_CACHE_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app

@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, gpu="A10G", timeout=300)
def go():
    import time
    t0 = time.time()
    from transformers import AutoTokenizer, AutoModel
    print(f"import done {time.time()-t0:.1f}s")
    tok = AutoTokenizer.from_pretrained("google/muril-base-cased", cache_dir=HF_CACHE_DIR)
    print(f"tokenizer loaded {time.time()-t0:.1f}s")
    model = AutoModel.from_pretrained("google/muril-base-cased", cache_dir=HF_CACHE_DIR)
    print(f"model loaded {time.time()-t0:.1f}s")
    import torch
    model = model.to("cuda")
    print(f"moved to cuda {time.time()-t0:.1f}s")
    batch = tok(["hello world"], return_tensors="pt").to("cuda")
    out = model(**batch)
    print(f"forward pass done {time.time()-t0:.1f}s, shape={out.last_hidden_state.shape}")

@app.local_entrypoint()
def main_m3diag():
    go.remote()
