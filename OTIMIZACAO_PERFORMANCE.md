# Guia de Otimização de Performance

## Problema: Geração de Imagens Muito Lenta

Se a geração de imagens está levando 10-15 minutos por imagem, você está usando **CPU** em vez de **GPU**.

## Verificar qual dispositivo está sendo usado

Ao executar a aplicação, você verá uma mensagem como:
```
============================================================
Carregando modelo runwayml/stable-diffusion-v1-5
Dispositivo: CPU
⚠️  AVISO: Usando CPU - a geração será MUITO lenta (10-15 min/imagem)
   Para acelerar, instale PyTorch com CUDA se tiver GPU NVIDIA
============================================================
```

## Soluções para Acelerar

### Opção 1: Usar GPU NVIDIA (Recomendado - 30x mais rápido)

Se você tem uma placa de vídeo NVIDIA:

1. **Verificar se tem GPU NVIDIA compatível:**
   ```bash
   nvidia-smi
   ```
   Se o comando funcionar, você tem GPU NVIDIA instalada.

2. **Instalar PyTorch com CUDA:**
   ```bash
   # Desinstalar PyTorch atual
   pip uninstall torch torchvision

   # Instalar versão com CUDA 11.8 (recomendado)
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Reiniciar a aplicação**
   ```bash
   streamlit run app.py
   ```

4. **Verificar se está usando GPU:**
   Você deve ver:
   ```
   Dispositivo: CUDA
   ✓ Usando GPU - geração rápida (~30 seg/imagem)
   ```

### Opção 2: Reduzir Qualidade para CPU (Se não tiver GPU)

Se não tiver GPU NVIDIA, você pode reduzir o número de passos de inferência para acelerar:

**No arquivo `app.py`, linha 233:**
```python
# Atual (50 passos - alta qualidade, 15 min)
num_inference_steps=50,

# Rápido (20 passos - qualidade OK, 6-8 min)
num_inference_steps=20,

# Muito rápido (10 passos - qualidade baixa, 3-4 min)
num_inference_steps=10,
```

### Opção 3: Usar Modelos Menores

Trocar para um modelo menor e mais rápido:

**No arquivo `app.py`, linha ~150:**
```python
# Atual (1.5 GB)
generator = ImageGenerator(model_id="runwayml/stable-diffusion-v1-5")

# Mais rápido (menor)
generator = ImageGenerator(model_id="CompVis/stable-diffusion-v1-4")
```

### Opção 4: Reduzir Resolução

**No arquivo `app.py`, linha 234-235:**
```python
# Atual (512x512)
width=512,
height=512,

# Mais rápido (256x256)
width=256,
height=256,
```

## Tabela de Performance

| Configuração | GPU NVIDIA | CPU |
|--------------|-----------|-----|
| Qualidade Alta (50 passos, 512x512) | ~30 seg | ~15 min |
| Qualidade Média (20 passos, 512x512) | ~15 seg | ~6 min |
| Qualidade Baixa (10 passos, 256x256) | ~8 seg | ~2 min |

## Testar GPU

Para verificar se o PyTorch está vendo sua GPU:

```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Nome da GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

Execute no terminal:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Links Úteis

- [Instalar CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch - Instalação com CUDA](https://pytorch.org/get-started/locally/)
- [Verificar compatibilidade GPU](https://developer.nvidia.com/cuda-gpus)

## Resumo

- **Tem GPU NVIDIA?** → Instale PyTorch com CUDA (30x mais rápido)
- **Sem GPU?** → Reduza qualidade (passos ou resolução)
- **Testando?** → Use `num_inference_steps=10` para testes rápidos
