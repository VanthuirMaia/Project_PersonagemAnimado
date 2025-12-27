# 🎬 Guia de Uso do Stable Video Diffusion (SVD)

## O que é o SVD?

O **Stable Video Diffusion (SVD)** é um modelo de IA que anima imagens estáticas, criando vídeos com movimento realista a partir de uma única imagem.

## 📍 Onde Encontrar o SVD na Interface?

### Passo 1: Abra o Streamlit
Execute a aplicação e abra no navegador:
```bash
.\run_app.ps1
# ou
.\run_app.bat
```

### Passo 2: Configure o Método de Animação

1. Na **barra lateral (sidebar)**, role até a seção **"Geração de Vídeo"**
2. No campo **"Método de Animação"**, selecione:
   ```
   "IA - Stable Video Diffusion"
   ```

### Passo 3: Configure os Parâmetros SVD

Após selecionar SVD, você verá as seguintes opções:

- **Resolução**: 
  - `512x320` (Recomendado 8GB) - Padrão
  - `384x256` (Ultra-Econômico) - Para GPUs com pouca memória
  - `640x384` (Avançado) - Para GPUs com muita memória

- **Frames do Vídeo**: 15-25 frames (padrão: 20)
  - Mais frames = vídeo mais longo, mas consome mais memória

- **FPS do Vídeo**: 3-7 fps (padrão: 4)
  - Menor FPS = vídeo mais longo

- **Passos de Inferência**: 20-30 (padrão: 25)
  - Mais passos = melhor qualidade, mas mais lento

### Passo 4: Gere as Imagens Primeiro

**IMPORTANTE**: O SVD anima **uma imagem por vez**, então você precisa gerar imagens primeiro:

1. Vá para a aba **"Geração"**
2. Descreva seu personagem
3. Clique em **"Gerar Imagens"**
4. Aguarde a geração concluir

### Passo 5: Animar com SVD

1. Vá para a aba **"Vídeo"**
2. Se você tem múltiplas imagens, escolha qual imagem animar:
   - Use o seletor **"Selecione a imagem:"**
   - Você verá uma prévia da imagem selecionada
3. Clique em **"🎬 Gerar Vídeo"**
4. Aguarde o processamento (2-3 minutos)

## ⚙️ Requisitos

### ✅ Obrigatório:
- **GPU NVIDIA com CUDA** (o SVD não funciona em CPU)
- Pelo menos **6-8 GB de VRAM** (memória da GPU)
- **Imagem gerada** pelo sistema

### ⚠️ Avisos:
- Na primeira execução, o modelo SVD será baixado automaticamente (~5GB)
- O processo pode demorar 2-5 minutos dependendo da GPU
- Se aparecer erro de memória, reduza a resolução ou número de frames

## 🔧 Resolução de Problemas

### Erro: "GPU CUDA não disponível"
- O SVD requer GPU NVIDIA
- Verifique se o PyTorch está instalado com suporte CUDA
- Use "Transições (OpenCV)" como alternativa (funciona em CPU)

### Erro: "Memória GPU insuficiente"
Tente:
1. Reduzir resolução para `384x256`
2. Reduzir frames para 15
3. Reduzir steps para 20
4. Fechar outros programas usando GPU

### Modelo não baixa
- Verifique sua conexão com a internet
- O modelo é baixado automaticamente na primeira vez
- O download pode levar vários minutos dependendo da velocidade

## 💡 Dicas

1. **Melhor Qualidade**: Use resolução maior e mais frames
2. **Economia de Memória**: Use `384x256` com 15 frames
3. **Vídeos Mais Longos**: Aumente o número de frames (até 25)
4. **Vídeos Mais Rápidos**: Reduza os passos de inferência (20)

## 🎯 Diferença Entre SVD e OpenCV

| Recurso | SVD (IA) | OpenCV (Transições) |
|---------|----------|---------------------|
| **Tipo** | Anima imagem individual | Combina múltiplas imagens |
| **Qualidade** | Movimento realista | Transições simples |
| **Requer GPU** | ✅ Sim | ❌ Não |
| **Tempo** | 2-5 minutos | ~30 segundos |
| **Memória** | 6-8 GB VRAM | Baixa |

## 📂 Onde os Vídeos São Salvos?

Os vídeos gerados com SVD são salvos em:
```
outputs/videos/svd_animation_YYYYMMDD_HHMMSS.mp4
```

Os metadados são salvos em:
```
outputs/videos/svd_metadata.json
```

