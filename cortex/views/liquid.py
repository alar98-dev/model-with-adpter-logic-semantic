import sys

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
            
            # --- O EVENTO INTERIOR ---
            if energy > 2.2 and synaptic_chunks:
                chunk = synaptic_chunks[chunk_idx % len(synaptic_chunks)]
                chunk_idx += 1
                
                if val_clamped < 0:
                    color = f"{global_modifier}\033[96;1m"
                else:
                    color = f"{global_modifier}\033[91;1m"
                    
                line += f"{color}{chunk}\033[0m"
                c += len(chunk)
                continue

            if energy < 0.1:
                ch = " "
                color = ""
            else:
                density_map = " `sey|oUh$qku<l}vb#t?ar&gcp>.ZwmQznj{ RXdNY;\\*Ji@W]ExFVS:/H,DBG%~KO'L+=MI(fT\"[PAC0)4758_6!9^3-21"
                map_idx = min(len(density_map) - 1, int((energy / 3.0) * len(density_map)))
                ch = density_map[map_idx]

                if val_clamped < -1.5:
                    color = f"{global_modifier}\033[36m" 
                elif val_clamped < 0:
                    color = f"{global_modifier}\033[34m" 
                elif val_clamped < 1.5:
                    color = f"{global_modifier}\033[35m" 
                else:
                    color = f"{global_modifier}\033[31m" 
                    
            if col_idx == 32:
                ch = "|"
                color = "\033[97;1m" # Branco brilhante para o Obelisco Divisor
            
            line += f"{color}{ch}\033[0m"
            c += 1
        
        sys.stdout.write(line + "\033[K\r\n")
        
    return chunk_idx
