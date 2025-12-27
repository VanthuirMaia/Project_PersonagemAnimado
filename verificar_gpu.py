"""
Script para verificar disponibilidade de GPU e CUDA
"""
import torch
import sys
import os

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

print("="*60)
print("VERIFICACAO DE GPU E CUDA")
print("="*60)

print(f"\nVersao do PyTorch: {torch.__version__}")
print(f"CUDA disponivel: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"[OK] GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"[OK] Versao CUDA: {torch.version.cuda}")
    print(f"[OK] Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n[OK] Tudo OK! A GPU esta pronta para uso.")
else:
    print("\n[ERRO] GPU nao detectada")
    print("\nPossiveis causas:")
    print("1. PyTorch instalado sem suporte CUDA")
    print("2. Driver NVIDIA nao instalado ou desatualizado")
    print("3. CUDA Toolkit nao instalado")

    print("\n" + "="*60)
    print("SOLUCAO")
    print("="*60)
    print("\nExecute os seguintes comandos:")
    print("\n1. Desinstalar PyTorch atual:")
    print("   pip uninstall torch torchvision -y")
    print("\n2. Instalar PyTorch com CUDA 11.8:")
    print("   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118")
    print("\n3. Verificar novamente:")
    print("   python verificar_gpu.py")

print("\n" + "="*60)
