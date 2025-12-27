"""
Script para verificar se CUDA está sendo utilizada no projeto
Verifica PyTorch, dispositivos, e uso de CUDA nos módulos principais
"""

import sys
import os
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass

print("="*70)
print("VERIFICAÇÃO DE CUDA NO PROJETO")
print("="*70)

# 1. Verificar Python e ambiente
print("\n1. INFORMAÇÕES DO AMBIENTE")
print("-" * 70)
print(f"Python: {sys.executable}")
print(f"Versão Python: {sys.version.split()[0]}")

# Verificar se está em ambiente virtual
venv_path = Path(sys.executable).parent.parent
if 'venv' in str(sys.executable).lower() or '.venv' in str(sys.executable):
    print(f"✅ Ambiente virtual detectado: {venv_path}")
else:
    print("⚠️  Não está em ambiente virtual (usando Python global)")

# 2. Verificar PyTorch
print("\n2. VERIFICAÇÃO DO PYTORCH")
print("-" * 70)
try:
    import torch
    print(f"✅ PyTorch instalado: {torch.__version__}")
    
    # Verificar se é versão CPU-only
    if '+cpu' in torch.__version__:
        print("⚠️  PyTorch CPU-only detectado (sem suporte CUDA)")
        print("   Para usar CUDA, instale PyTorch com CUDA:")
        print("   pip uninstall torch torchvision -y")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    else:
        print(f"✅ PyTorch com suporte CUDA")
        
except ImportError:
    print("❌ PyTorch não instalado")
    print("   Instale com: pip install torch torchvision")
    sys.exit(1)

# 3. Verificar CUDA
print("\n3. VERIFICAÇÃO DE CUDA")
print("-" * 70)
cuda_available = torch.cuda.is_available()
print(f"CUDA disponível: {cuda_available}")

if cuda_available:
    try:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU detectada: {gpu_name}")
        print(f"✅ Versão CUDA: {cuda_version}")
        print(f"✅ Memória GPU: {gpu_memory:.2f} GB")
        
        # Testar criação de tensor na GPU
        try:
            test_tensor = torch.zeros(1).cuda()
            print(f"✅ Tensor criado na GPU: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ CUDA funcionando perfeitamente!")
        except Exception as e:
            print(f"⚠️  Erro ao criar tensor na GPU: {e}")
            
    except Exception as e:
        print(f"❌ Erro ao acessar GPU: {e}")
else:
    print("❌ CUDA não disponível")
    print("\nPossíveis causas:")
    print("  1. PyTorch instalado sem suporte CUDA (versão CPU-only)")
    print("  2. Driver NVIDIA não instalado ou desatualizado")
    print("  3. CUDA Toolkit não instalado")
    print("  4. GPU não compatível com CUDA")

# 4. Verificar módulos do projeto
print("\n4. VERIFICAÇÃO DOS MÓDULOS DO PROJETO")
print("-" * 70)

# Verificar image_generator
try:
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from image_generator import ImageGenerator
    
    print("✅ image_generator.py importado com sucesso")
    
    # Testar inicialização
    try:
        generator = ImageGenerator(device="auto")
        print(f"✅ ImageGenerator inicializado")
        print(f"   Dispositivo selecionado: {generator.device}")
        
        if generator.device == "cuda":
            print("   ✅ Usando CUDA para geração de imagens")
        else:
            print("   ⚠️  Usando CPU (muito lento)")
            
    except Exception as e:
        print(f"⚠️  Erro ao inicializar ImageGenerator: {e}")
        
except ImportError as e:
    print(f"⚠️  Erro ao importar image_generator: {e}")

# Verificar video_generator
try:
    from video_generator import VideoGenerator
    
    print("✅ video_generator.py importado com sucesso")
    
    # Verificar se SVD requer CUDA
    if cuda_available:
        print("   ✅ CUDA disponível para Stable Video Diffusion")
    else:
        print("   ⚠️  CUDA não disponível - SVD não funcionará")
        print("   (mas transições OpenCV funcionarão em CPU)")
        
except ImportError as e:
    print(f"⚠️  Erro ao importar video_generator: {e}")

# 5. Resumo final
print("\n5. RESUMO")
print("-" * 70)

if cuda_available:
    print("✅ CUDA está sendo utilizada no projeto")
    print("   - Geração de imagens: GPU (rápido)")
    print("   - Geração de vídeo SVD: GPU (disponível)")
    print("   - Geração de vídeo OpenCV: CPU/GPU (sempre disponível)")
else:
    print("⚠️  CUDA não está disponível")
    print("   - Geração de imagens: CPU (muito lento - ~5-10 min/imagem)")
    print("   - Geração de vídeo SVD: Não disponível (requer GPU)")
    print("   - Geração de vídeo OpenCV: CPU (disponível)")
    print("\n💡 Para usar CUDA:")
    print("   1. Instale PyTorch com CUDA:")
    print("      pip uninstall torch torchvision -y")
    print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("   2. Execute este script novamente para verificar")

print("\n" + "="*70)

