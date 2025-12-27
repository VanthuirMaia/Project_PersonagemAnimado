# Início Rápido

Guia de 5 minutos para começar a usar o sistema.

**Projeto**: Personagem Generativo e Animação Curta
**Instituição**: Universidade de Pernambuco (UPE)
**Programa**: Residência em IA Generativa
**Disciplina**: IA Generativa para Mídia Visual
**Autores**: Vanthuir Maia e Rodrigo Santana

---

## 1. Instalação (3 minutos)

### Windows

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### Linux/Mac

```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## 2. Executar (30 segundos)

```bash
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`

## 3. Gerar seu Personagem (2 minutos)

### Passo 1: Descrever
Na aba **Geração**, digite uma descrição detalhada:

**Exemplo**:
```
A cute blue dragon mascot, chibi style, big round eyes,
small wings, friendly smile, cartoon style, simple design,
white background, digital art, consistent character
```

### Passo 2: Configurar (Opcional)
Na barra lateral esquerda:
- **Número de Imagens**: 10 (mínimo do projeto)
- **Seed**: 42 (ou qualquer número para reproduzir)
- **Guidance Scale**: 7.5 (aumentar para mais fidelidade ao prompt)

### Passo 3: Gerar
Clique em **"Gerar Imagens"**

⏱️ Tempo: 10-15 minutos (GPU) ou 60+ minutos (CPU)

### Passo 4: Visualizar
Vá para aba **Imagens** para ver a galeria

## 4. Criar Vídeo (1 minuto)

### Passo 1: Configurar
Na barra lateral:
- **FPS**: 10 (velocidade do vídeo)
- **Duração por Imagem**: 0.5s (para vídeo curto)
- **Frames de Transição**: 5
- **Adicionar Loop**: ✅ Marcado

### Passo 2: Gerar
Na aba **Vídeo**, clique em **"Gerar Vídeo"**

⏱️ Tempo: ~30 segundos

### Passo 3: Assistir e Baixar
O vídeo será exibido automaticamente. Clique em **"Download do Vídeo"** para salvar.

## 5. Exportar Documentação

Na aba **Documentação**:
- Leia sobre o pipeline técnico
- Clique em **"Download Documentação (JSON)"** para metadados completos

---

## Dicas Rápidas

### Para vídeos de 5-20 segundos:
```
FPS = 10
Duração por Imagem = 0.5
Frames de Transição = 5
→ Resultado: ~12 segundos
```

### Para melhor consistência:
- Use prompts MUITO detalhados
- Mencione "consistent character design"
- Use a mesma seed base
- Evite palavras ambíguas

### Se der erro de memória (GPU):
- Reduza número de imagens (gere 5 por vez)
- Feche outros programas
- Reinicie o Streamlit

### Primeira execução é lenta:
- O modelo (~5GB) será baixado automaticamente
- Aguarde o download completar
- Execuções seguintes serão mais rápidas

---

## Exemplos de Prompts

### Mascote Corporativo
```
A professional penguin mascot wearing a blue tie,
business casual style, friendly expression,
standing confidently, modern flat design,
white background, corporate branding, clean lines
```

### Criatura Fantástica
```
A mystical forest spirit, glowing green eyes,
made of leaves and flowers, whimsical fantasy style,
cute and friendly, magical aura, ethereal lighting,
concept art, high quality
```

### Avatar Gamer
```
A cyberpunk hacker character, neon purple hair,
futuristic visor, tech jacket, confident pose,
anime style, vibrant colors, digital art,
character design sheet
```

---

## Resolução de Problemas

### Erro: "No module named 'diffusers'"
```bash
pip install -r requirements.txt
```

### Erro: "CUDA out of memory"
- Gere menos imagens por vez
- Use configuração "Rápido" (30 steps)

### Vídeo não abre
- Certifique-se de ter codec MP4 instalado
- Tente outro player de vídeo
- Formato é MP4 padrão (H.264)

### Geração muito lenta
- Verifique se está usando GPU (veja mensagem no terminal)
- Para forçar CPU: edite `device="cpu"` em image_generator.py
- Considere usar menos passos (30-40)

---

## Próximos Passos

Depois de criar seu primeiro personagem:

1. Experimente diferentes prompts
2. Varie os parâmetros para ver o efeito
3. Leia `DOCUMENTACAO_TECNICA.md` para entender o pipeline
4. Explore o código em `src/` para customizações
5. Compartilhe seus resultados!

---

**Precisa de ajuda?** Consulte `README.md` para documentação completa.
