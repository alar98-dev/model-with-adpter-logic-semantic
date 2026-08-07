import uuid
import re

class QuantumMACEncoder:
    def __init__(self):
        pass

    def extrair_assinatura_fisica(self):
        """
        Captura o endereço MAC primário do Hardware.
        Retorna no formato padronizado (ex: 00:11:22:33:44:55).
        """
        mac_int = uuid.getnode()
        mac_hex = ':'.join(re.findall('..', '%012x' % mac_int)).upper()
        return mac_hex

    def ancorar_no_cortex(self, weight_library):
        """
        Transforma a assinatura MAC em vetor 640D na IDKB.
        """
        mac = self.extrair_assinatura_fisica()
        weight_library.ancorar_dna_hardware(mac)
        return mac

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from vector_weight_library import WeightKnowledgeLibrary
    
    library = WeightKnowledgeLibrary(vector_dim=640)
    encoder = QuantumMACEncoder()
    mac = encoder.ancorar_no_cortex(library)
    print(f"✅ DNA Físico injetado com sucesso! MAC: {mac}")
