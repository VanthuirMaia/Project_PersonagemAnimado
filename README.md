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

1. Na aba **"Vídeo"**, ajuste os parâmetros:
   - FPS (frames por segundo)
   - Duração por imagem
   - Frames de transição
   - Ativar/desativar loop
2. Clique em **"Gerar Vídeo"**
3. Aguarde a criação do vídeo (geralmente rápido, ~30 segundos)
4. Assista ao vídeo e faça download se desejar

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

# Criar vídeo
video_path = video_gen.create_video_from_images(
    images=image_files,
    fps=3,
    duration_per_image=1.5,
    transition_frames=15,
    add_loop=True
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

**Biblioteca**: OpenCV (cv2)

**Técnica**: Interpolação linear entre frames (cross-dissolve)

**Processo**:

1. Cada imagem é mantida por N frames estáticos
2. Transições suaves usando `cv2.addWeighted`
3. Loop opcional para animação contínua

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

- **Desafio**: Transições suaves
- **Solução Atual**: Interpolação linear
- **Melhoria Futura**: Motion transfer, text-to-video models

### Recursos Computacionais

- Geração local requer GPU com boa memória
- Em CPU, a geração é muito mais lenta
- Modelos ocupam ~5GB de espaço

## Melhorias Futuras

1. **ControlNet**: Maior controle sobre pose e estrutura
2. **Motion Transfer**: MediaPipe Pose para animações mais naturais
3. **Text-to-Video**: Integração com Gen-2, Pika Labs, Runway
4. **Efeitos**: Zoom, pan, rotate nas transições
5. **API Integration**: Suporte para APIs cloud (Stability AI, Replicate)

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
