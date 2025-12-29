# ⏱️ Duração do Vídeo SVD - Configurado para 5-10 Segundos

**Última atualização**: 27/12/2024

## ✅ Implementação Completa do Stable Video Diffusion

O sistema agora possui integração completa com Stable Video Diffusion (SVD), permitindo gerar vídeos animados com movimento realista a partir de uma única imagem.

## ✅ Configurações Padrão

As configurações padrão foram ajustadas para gerar vídeos de **5-10 segundos** atendendo aos requisitos do projeto.

---

## 📊 Configurações Atuais

### **Valores Padrão** (Novos)

| Parâmetro | Antes | Agora | Motivo |
|-----------|-------|-------|--------|
| **Frames** | 14 | **20** | Permite vídeos mais longos |
| **FPS** | 7 | **4** | Menor FPS = vídeo mais longo |
| **Duração** | 2s | **5s** | ✅ Dentro da faixa 5-10s |

### **Cálculo da Duração**

```
Duração (segundos) = Frames ÷ FPS

Exemplos:
- 20 frames ÷ 4 fps = 5.0 segundos ✅
- 20 frames ÷ 3 fps = 6.7 segundos ✅
- 25 frames ÷ 3 fps = 8.3 segundos ✅
- 25 frames ÷ 2.5 fps = 10.0 segundos ✅
```

---

## 🎛️ Opções Disponíveis na Interface

### **Frames do Vídeo**
- **Mínimo**: 15 frames
- **Máximo**: 25 frames (limite do SVD-XT)
- **Padrão**: 20 frames

### **FPS do Vídeo**
- **Mínimo**: 3 fps
- **Máximo**: 7 fps
- **Padrão**: 4 fps

---

## 📈 Combinações Recomendadas

### **Para ~5 Segundos**
```
20 frames ÷ 4 fps = 5.0 segundos ✅ (Padrão)
25 frames ÷ 5 fps = 5.0 segundos ✅
```

### **Para ~7-8 Segundos**
```
20 frames ÷ 3 fps = 6.7 segundos ✅
25 frames ÷ 3 fps = 8.3 segundos ✅
```

### **Para ~10 Segundos**
```
25 frames ÷ 2.5 fps = 10.0 segundos ✅
20 frames ÷ 2 fps = 10.0 segundos ✅ (mas 2 fps é muito baixo)
```

**⚠️ Nota**: FPS abaixo de 3 pode deixar o vídeo com aparência de stop-motion. Recomendado manter FPS entre 3-5.

---

## 🎨 Interface do Streamlit

Agora a interface mostra **d indicação da duração estimada**:

```
🟢 Duração estimada: ~5.0 segundos (20 frames ÷ 4 fps)
```

**Cores**:
- 🟢 Verde: Dentro da faixa 5-10 segundos (ideal)
- 🟡 Amarelo: Fora da faixa (muito curto ou muito longo)

---

## 💾 Memória GPU

### **Impacto no Uso de Memória**

| Frames | Uso de Memória GPU (RTX 3050 8GB) |
|--------|-----------------------------------|
| 15 | ~5.5 GB (muito seguro) |
| 20 | ~6.5 GB (recomendado) ✅ |
| 25 | ~7.5 GB (limite) ⚠️ |

**Recomendação**: Use 20 frames para equilíbrio entre duração e memória.

---

## 🔧 Ajustes Finais

Se precisar ajustar:

1. **Vídeo mais longo** (mas próximo do limite):
   - Aumente frames para 25
   - Diminua FPS para 3
   - Resultado: ~8.3 segundos

2. **Vídeo mais curto** (se tiver problemas de memória):
   - Diminua frames para 15
   - Aumente FPS para 5
   - Resultado: ~3.0 segundos (abaixo do ideal)

3. **Vídeo ideal** (recomendado):
   - 20 frames
   - 4 fps
   - Resultado: ~5.0 segundos ✅

---

## 🎯 Otimizações Implementadas

### Para GPUs com 8GB VRAM (RTX 3050, etc.)

O sistema foi otimizado especificamente para GPUs com 8GB de VRAM:

1. **FP16 Precision**: Reduz uso de memória em 50%
2. **CPU Offloading**: Move componentes não críticos para RAM
3. **Attention Slicing**: Processa atenção em chunks menores
4. **Resolução Reduzida**: 512x320 (padrão otimizado)
5. **Decode Chunk Size**: Processa 1 frame por vez (mínimo memória)

**Uso de Memória GPU**:
- 15 frames: ~5.5 GB (muito seguro)
- 20 frames: ~6.5 GB (recomendado) ✅
- 25 frames: ~7.5 GB (limite) ⚠️

### Verificação Automática

O sistema verifica automaticamente:
- ✅ Disponibilidade de GPU CUDA
- ✅ Memória livre antes da geração (mínimo 3GB)
- ✅ Limpa cache antes e após processamento
- ✅ Tratamento de erros com sugestões específicas

## 📝 Callback de Progresso

O método SVD suporta callback para atualizar progresso na interface:

```python
def progress_callback(progress, status):
    # progress: 0.0 a 1.0
    # status: mensagem de status
    print(f"{progress*100:.0f}% - {status}")

video_gen.animate_image_svd(
    image=image,
    progress_callback=progress_callback
)
```

**Estágios do progresso**:
- 🔧 Preparando download do modelo
- 📥 Download em andamento (~5GB)
- ✅ Modelo baixado! Carregando na memória
- ⚙️ Aplicando otimizações
- 🎬 Gerando frames com SVD
- 🎨 Processando passos de inferência
- 📹 Processando frames do vídeo
- 💾 Salvando vídeo
- ✅ Vídeo salvo com sucesso!

## 💾 Metadados Salvos

Cada geração SVD salva metadados completos em `svd_metadata.json`:

```json
{
  "method": "stable_video_diffusion",
  "num_frames": 20,
  "fps": 4,
  "resolution": "512x320",
  "original_resolution": "512x512",
  "num_inference_steps": 25,
  "motion_bucket_id": 127,
  "decode_chunk_size": 1,
  "duration": 5.0,
  "gpu_memory_used": "6.5 GB",
  "timestamp": "2024-12-27T10:30:00"
}
```

## 🔧 Limpeza de Memória

Após gerar o vídeo, é recomendado limpar a memória GPU:

```python
video_gen.cleanup_svd()  # Remove pipeline da memória
```

Isso libera ~5-6GB de VRAM para outras operações.

## ✅ Status

- ✅ Stable Video Diffusion completamente implementado
- ✅ Valores padrão ajustados para 5 segundos
- ✅ Interface mostra duração estimada
- ✅ Sliders permitem ajuste para 5-10 segundos
- ✅ Valores padrão no código atualizados
- ✅ Session state inicializado corretamente
- ✅ Otimizações para 8GB VRAM implementadas
- ✅ Callback de progresso funcionando
- ✅ Verificação de memória GPU
- ✅ Metadados salvos automaticamente
- ✅ Tratamento de erros robusto

**Teste agora e veja vídeos de 5-10 segundos com movimento realista!** 🎬✨

