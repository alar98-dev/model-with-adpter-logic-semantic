import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)

class AeonTopologicalKernel(nn.Module):
    """
    O Motor Topológico Nativo (CNN Residual).
    Opera inteiramente na física matricial geométrica (1 canal x 64px x 64px). 
    Sem tokens. Sem linguagem. Pura predição e transmutação de estado magnético.
    """
    def __init__(self, hidden_dim=64, num_blocks=4):
        super().__init__()
        # Recebe a topologia do espaço 64x64 e expande a energia para múltiplos canais (dimensões ocultas)
        self.entry_conv = nn.Conv2d(1, hidden_dim, kernel_size=5, padding=2)
        
        # O processamento biológico profundo (difusão de vizinhança geométrica)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        
        # A convolução final comprime a "ideia abstrata" de volta para um único canal tensional físico (A tela final)
        self.exit_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: [Batch, 1, 64, 64]
        out = F.gelu(self.entry_conv(x))
        out = self.res_blocks(out)
        out = self.exit_conv(out)
        
        # O tensor de saída é o "Próximo Pensamento" ou "O Pensamento Refinado"
        return out

    def energy_loss(self, predicted_state, actual_next_state):
        """
        A perda de Energia (Tensão Geométrica).
        Avalia a distância Euclidiana e a fidelidade da forma topológica entre o pensamento gerado e o pensamento alvo.
        """
        mse = F.mse_loss(predicted_state, actual_next_state)
        # Cosine Loss na grade achatada para garantir alinhamento estrutural, não apenas valor absoluto
        cos_loss = 1 - F.cosine_similarity(
            predicted_state.view(predicted_state.size(0), -1),
            actual_next_state.view(actual_next_state.size(0), -1),
            dim=1
        ).mean()
        
        return mse + (cos_loss * 0.5)

if __name__ == "__main__":
    print("\033[96m[AeonKernel]\033[0m Inicializando Motor Topológico (Resoluição Nativa 64x64)...")
    kernel = AeonTopologicalKernel(hidden_dim=64, num_blocks=5)
    
    # Injetando uma forma aleatória para simular o vácuo respirando
    caos_input = torch.randn(1, 1, 64, 64)
    print(f"Dimensão de Entrada: {caos_input.shape}")
    
    next_thought = kernel(caos_input)
    print(f"Dimensão de Saída (Próximo Pensamento): {next_thought.shape}")
    
    print("\033[92mO núcleo convolucional operou o tensor nativamente sem LLMs.\033[0m")
