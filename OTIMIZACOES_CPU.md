# Otimizações para Rodar em CPU

## Problema Identificado

Seu sistema está rodando em **CPU** (sem GPU CUDA), o que torna a geração muito lenta:
- **Com GPU**: ~10-15 minutos para 10 imagens
- **Com CPU**: ~60-120 minutos para 10 imagens

## Soluções Rápidas

### 1. Use Configurações Ultra Rápidas

Na interface Streamlit, configure:

```
Número de Imagens: 5 (em vez de 10)
Passos de Inferência: 20 (em vez de 50)
Guidance Scale: 7.0
```

**Tempo estimado**: ~20-30 minutos

### 2. Teste com 1 Imagem Primeiro

Para testar se está funcionando:
- Coloque **1 imagem**
- **20 passos**
- Aguarde ~3-5 minutos

### 3. Use Resolução Menor (Futuro)

Editar `src/image_generator.py` linha 120:
```python
width=256,   # em vez de 512
height=256   # em vez de 512
```

**Ganho**: 4x mais rápido

## Alternativas Recomendadas

### Opção A: Usar APIs Cloud (Mais Rápido)

Se tiver conta em serviços de IA:

1. **Replicate** (https://replicate.com)
   - Grátis: ~100 gerações/mês
   - Rápido: ~10 segundos por imagem

2. **Hugging Face Inference API**
   - Grátis com limitações
   - 30 segundos por imagem

### Opção B: Google Colab (GPU Grátis)

1. Acesse: https://colab.research.google.com
2. Habilite GPU gratuita
3. Execute o código lá
4. Download das imagens

### Opção C: Rodar Overnight

Se não tiver pressa:
- Configure para gerar 10 imagens
- Deixe rodando durante a noite
- Pela manhã estará pronto

## Monitoramento

Durante a geração, você verá no terminal:
```
Gerando imagem 1/10 (seed: 42)...
  Salva: character_001_seed42.png
```

**Cada imagem leva ~5-10 minutos em CPU**

## Comparação de Tempo

| Configuração | GPU (RTX 3060) | CPU (i7) |
|--------------|----------------|----------|
| 1 img, 20 steps | 15s | 3 min |
| 1 img, 50 steps | 30s | 6 min |
| 10 imgs, 20 steps | 3 min | 30 min |
| 10 imgs, 50 steps | 8 min | 90 min |

## Dica: Barra de Progresso

O Streamlit mostra "Gerando imagens..." mas você pode acompanhar o progresso detalhado no **terminal/console** onde executou `streamlit run app.py`.

## Para Projetos Futuros

Considere:
- Usar serviços cloud para geração (Replicate, Stability AI)
- Investir em GPU (mesmo uma entrada como GTX 1660 ajuda muito)
- Usar Google Colab para projetos acadêmicos
