import sys
import random

def render_view(chaos_norm, max_rows, max_cols, cols, synaptic_chunks, global_modifier, chunk_idx):
    for r in range(max_rows):
        row_idx = min(63, int((r / max_rows) * 64))
        line = ""
        c = 0
        while c < max_cols:
            col_idx = min(63, int((c / max_cols) * 64))
            val = chaos_norm[row_idx, col_idx]
            
            val_clamped = max(-3.0, min(3.0, val))
            energy = abs(val_clamped)
            
            # --- O EVENTO INTERIOR (No Plasma, chunks são despedaçados brutalmente) ---
            if energy > 1.8 and synaptic_chunks:
                chunk = synaptic_chunks[chunk_idx % len(synaptic_chunks)]
                chunk_idx += 1
                
                # Glitch no chunk
                if random.random() > 0.7:
                    chunk = "".join(random.sample(chunk, len(chunk)))
                    
                color = f"{global_modifier}\033[91;5m" if random.random() > 0.5 else f"{global_modifier}\033[95;1m"
                line += f"{color}{chunk}\033[0m"
                c += len(chunk)
                continue

            # Plasma tem buracos aleatórios
            if energy < 0.2 and random.random() > 0.1:
                ch = " "
                color = ""
            else:
                plasma_map = "X%#@*!^&~?><;:|/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                if energy < 0.1:
                    ch = "."
                    color = f"{global_modifier}\033[90m"
                else:
                    map_idx = min(len(plasma_map) - 1, int((energy / 3.0) * len(plasma_map)))
                    ch = plasma_map[map_idx]

                    if val_clamped < -1.5:
                        color = f"{global_modifier}\033[91;1m" 
                    elif val_clamped < 0:
                        color = f"{global_modifier}\033[31m" 
                    elif val_clamped < 1.5:
                        color = f"{global_modifier}\033[33m" 
                    else:
                        color = f"{global_modifier}\033[93;1m" 

            if col_idx == 32:
                ch = "|"
                color = "\033[97;1m"
            
            line += f"{color}{ch}\033[0m"
            c += 1
        
        sys.stdout.write(line + "\033[K\r\n")
        
    return chunk_idx
