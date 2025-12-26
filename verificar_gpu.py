"""
Script para verificar disponibilidade de GPU e CUDA
"""
import torch
import sys

print("="*60)
print("VERIFICAÇÃO DE GPU E CUDA")
print("="*60)

print(f"\nVersão do PyTorch: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"✓ Versão CUDA: {torch.version.cuda}")
    print(f"✓ Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n✓ Tudo OK! A GPU está pronta para uso.")
else:
    print("\n✗ GPU não detectada")
    print("\nPossíveis causas:")
    print("1. PyTorch instalado sem suporte CUDA")
    print("2. Driver NVIDIA não instalado ou desatualizado")
    print("3. CUDA Toolkit não instalado")

    print("\n" + "="*60)
    print("SOLUÇÃO")
    print("="*60)
    print("\nExecute os seguintes comandos:")
    print("\n1. Desinstalar PyTorch atual:")
    print("   pip uninstall torch torchvision -y")
    print("\n2. Instalar PyTorch com CUDA 11.8:")
    print("   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118")
    print("\n3. Verificar novamente:")
    print("   python verificar_gpu.py")

print("\n" + "="*60)
