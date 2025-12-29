# Personagem Generativo e Animação Curta

Pipeline completo de IA Generativa para criação de personagens visuais consistentes e geração de vídeo animado.

## Descrição do Projeto

Este projeto implementa um sistema de geração de personagens usando Stable Diffusion e criação de vídeos animados a partir das imagens geradas. O objetivo é criar:

- **Mínimo 10 imagens** do personagem com consistência visual
- **Vídeo animado** de 5-20 segundos preservando a identidade do personagem
- **Documentação técnica** completa do pipeline

## Estrutura do Projeto

```
projeto_PersonagemAnimado/
├── app.py                      # Interface Streamlit principal
├── src/
│   ├── image_generator.py      # Módulo de geração de imagens
│   └── video_generator.py      # Módulo de geração de vídeo
├── outputs/
│   ├── images/                 # Imagens geradas
│   └── videos/                 # Vídeos gerados
├── Docs/
│   └── Projeto da Disciplina.pdf
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- GPU com CUDA (recomendado) ou CPU (mais lento)
- 8GB+ de RAM (16GB recomendado)
- Espaço em disco: ~5GB para modelos

### Passo a Passo

1. **Clone ou baixe o projeto**

```bash
cd projeto_PersonagemAnimado
```

2. **Crie um ambiente virtual**

```bash
python -m venv .venv
```

3. **Ative o ambiente virtual**

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

4. **Instale as dependências**

```bash
pip install -r requirements.txt
```

**Nota**: A instalação pode levar alguns minutos, especialmente o PyTorch.

### Instalação do PyTorch com CUDA (Opcional mas Recomendado)

Se você tem uma GPU NVIDIA, instale a versão CUDA do PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Como Usar

### 1. Executar a Interface Streamlit

```bash
streamlit run app.py
```

O aplicativo abrirá no navegador em `http://localhost:8501`

### 2. Gerar Imagens do Personagem

1. Na aba **"Geração"**, descreva seu personagem no campo de prompt
2. Ajuste os parâmetros na barra lateral (opcional):
   - Número de imagens (mínimo 10)
   - Seed (para reprodutibilidade)
   - Guidance Scale (7-15 recomendado)
   - Passos de Inferência (50 é bom equilíbrio)
3. Clique em **"Gerar Imagens"**
4. Aguarde a geração (pode levar 5-15 minutos dependendo do hardware)

### 3. Visualizar Imagens

1. Vá para a aba **"Imagens"**
2. Visualize todas as imagens geradas em grade
3. Confira os parâmetros de geração

### 4. Criar Vídeo Animado

1. Na aba **"Vídeo"**, escolha o método:
   - **Transições (OpenCV)**: Para múltiplas imagens com fade (funciona em qualquer hardware)
   - **IA - Stable Video Diffusion**: Para animar uma imagem com movimento real (requer GPU)
2. Ajuste os parâmetros conforme o método escolhido:
   - **OpenCV**: FPS, duração por imagem, frames de transição, loop
   - **SVD**: Frames (15-25), FPS (3-7), resolução, passos de inferência
3. Se usar SVD, escolha qual imagem animar (método anima uma por vez)
4. Clique em **"Gerar Vídeo"**
5. Aguarde a criação:
   - OpenCV: ~30 segundos
   - SVD: 2-3 minutos (primeira vez baixa modelo)
6. Assista ao vídeo e faça download se desejar

### 5. Exportar Documentação

1. Vá para a aba **"Documentação"**
2. Leia sobre o pipeline técnico
3. Clique em **"Download Documentação"** para exportar metadados em JSON

## Uso via Scripts Python

### Gerar Imagens Diretamente

```python
from src.image_generator import ImageGenerator

# Criar gerador
generator = ImageGenerator()

# Gerar imagens
images = generator.generate_images(
    prompt="A cute cartoon robot, blue and white colors",
    num_images=10,
    seed=42,
    guidance_scale=7.5,
    num_inference_steps=50
)
```

### Criar Vídeo Diretamente

```python
from src.video_generator import VideoGenerator
from glob import glob

# Criar gerador de vídeo
video_gen = VideoGenerator()

# Buscar imagens
image_files = sorted(glob("outputs/images/*/character_*.png"))

# Método 1: Criar vídeo com transições (OpenCV)
video_path = video_gen.create_video_from_images(
    images=image_files,
    fps=3,
    duration_per_image=1.5,
    transition_frames=15,
    add_loop=True
)

# Método 2: Animar imagem com SVD (requer GPU)
from PIL import Image
image = Image.open("outputs/images/character_001.png")
video_path = video_gen.animate_image_svd(
    image=image,
    output_path="outputs/videos/svd_animation.mp4",
    num_frames=20,  # Para ~5 segundos
    fps=4,
    resolution=(512, 320),  # Otimizado para 8GB VRAM
    num_inference_steps=25
)
```

## Pipeline Técnico

### Geração de Imagens

**Modelo**: Stable Diffusion v1.5 (Hugging Face Diffusers)

**Estratégia de Consistência**:

- Seeds sequenciais a partir de uma seed base
- Prompt detalhado e consistente
- Negative prompt para evitar artefatos

**Parâmetros Principais**:

- `guidance_scale`: Controla aderência ao prompt (7-15)
- `num_inference_steps`: Qualidade da geração (30-100)
- `seed`: Reprodutibilidade

### Geração de Vídeo

O sistema oferece **duas abordagens** para criação de vídeo:

#### 1. Método de Transições (OpenCV)

**Biblioteca**: OpenCV (cv2)  
**Técnica**: Interpolação linear entre frames (cross-dissolve)  
**Requisitos**: Qualquer hardware (CPU ou GPU)

**Processo**:
1. Cada imagem é mantida por N frames estáticos
2. Transições suaves usando `cv2.addWeighted`
3. Loop opcional para animação contínua

**Vantagens**:
- ✅ Funciona em CPU ou GPU
- ✅ Rápido (~30 segundos)
- ✅ Não requer download adicional de modelos

**Limitações**:
- ⚠️ Apenas transições (fade), não movimento real
- ⚠️ Resultado é mais "slideshow" que animação

#### 2. Método Stable Video Diffusion (SVD) 🆕

**Modelo**: Stable Video Diffusion XT (Hugging Face)  
**Técnica**: IA generativa para animar imagens  
**Requisitos**: GPU CUDA com 8GB+ VRAM

**Processo**:
1. Anima uma única imagem com movimento realista
2. Gera vídeo de 5-10 segundos automaticamente
3. Preserva identidade visual da imagem

**Vantagens**:
- ✅ Movimento real gerado por IA
- ✅ Vídeos mais naturais e dinâmicos
- ✅ Preserva identidade visual perfeitamente

**Limitações**:
- ⚠️ Requer GPU CUDA
- ⚠️ Primeira execução baixa modelo grande (~5GB)
- ⚠️ Processamento mais lento (2-3 minutos)

**Otimizações Implementadas**:
- FP16 (metade da memória)
- CPU Offloading (move partes para RAM)
- Attention Slicing máximo
- Resolução otimizada (512x320)
- Suporta GPUs com apenas 8GB VRAM

## Ferramentas Utilizadas

- **diffusers**: Geração de imagens com Stable Diffusion
- **transformers**: Modelos de linguagem
- **torch**: Backend de deep learning
- **opencv-python**: Processamento de vídeo
- **streamlit**: Interface web
- **PIL/Pillow**: Manipulação de imagens

## Desafios e Limitações

### Consistência Visual

- **Desafio**: Manter identidade entre gerações
- **Solução Atual**: Seeds sequenciais
- **Melhoria Futura**: ControlNet com pose reference

### Coerência Temporal

- **Desafio**: Transições suaves e movimento real
- **Solução Atual**: 
  - Método 1: Interpolação linear (transições)
  - Método 2: Stable Video Diffusion (movimento real) ✅ **IMPLEMENTADO**
- **Melhoria Futura**: Motion transfer, integração com outros modelos text-to-video

### Recursos Computacionais

- **Geração de imagens**: Requer GPU para velocidade adequada (CPU é muito lento)
- **Vídeo OpenCV**: Funciona em qualquer hardware (CPU ou GPU)
- **Vídeo SVD**: Requer GPU CUDA com 8GB+ VRAM (otimizado para 8GB)
- **Espaço em disco**: 
  - Stable Diffusion: ~5GB
  - SVD: +5GB (baixado na primeira execução)
  - Total: ~10GB

## Melhorias Futuras

1. ✅ **Stable Video Diffusion**: Implementado - animação realista com IA
2. **ControlNet**: Maior controle sobre pose e estrutura
3. **Motion Transfer**: MediaPipe Pose para animações mais naturais
4. **Text-to-Video**: Integração com outros modelos (Gen-2, Pika Labs, Runway)
5. **Efeitos**: Zoom, pan, rotate nas transições
6. **API Integration**: Suporte para APIs cloud (Stability AI, Replicate)
7. **SVD Multi-Image**: Animar múltiplas imagens sequencialmente

## Requisitos do Projeto (Checklist)

- [x] Mínimo 10 imagens do personagem
- [x] Vídeo animado de 5-20 segundos
- [x] Pipeline de geração estruturado
- [x] Controle de parâmetros (seeds, guidance, prompts)
- [x] Consistência visual entre imagens
- [x] Preservação de identidade no vídeo
- [x] Documentação técnica
- [x] Scripts e códigos organizados
- [x] Metadados salvos (JSON)

## Troubleshooting

### Erro: "CUDA out of memory"

- Reduza o tamanho das imagens (width/height)
- Reduza batch size (gere menos imagens por vez)
- Use `use_fp16=True` para economizar memória
- Feche outros programas que usam GPU

### Erro: "Model not found"

- Verifique conexão com internet (primeiro uso baixa modelo)
- Aguarde download completar (~5GB)
- Verifique espaço em disco

### Geração muito lenta

- Use GPU em vez de CPU
- Reduza `num_inference_steps`
- Considere usar APIs cloud

## Autores

**Vanthuir Maia** - vanmaiasf@gmail.com
**Rodrigo Santana** - rodrigoalisson33@gmail.com

## Informações Acadêmicas

**Instituição**: Universidade de Pernambuco (UPE)
**Programa**: Residência em IA Generativa
**Disciplina**: IA Generativa para Mídia Visual

## Licença

Este projeto é para fins educacionais.
