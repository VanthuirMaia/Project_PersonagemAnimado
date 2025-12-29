"""
Interface Streamlit - Personagem Generativo e Animação Curta
Pipeline completo de geração de personagens e vídeo animado

Instituição: Universidade de Pernambuco (UPE)
Programa: Residência em IA Generativa
Disciplina: IA Generativa para Mídia Visual
Autores: Vanthuir Maia e Rodrigo Santana
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import shutil
import time
import os

# Importar torch normalmente (sem tentar limpar/reimportar)
try:
    import torch
    print(f"[INIT] PyTorch importado: {torch.__version__}")
    print(f"[INIT] PyTorch localização: {torch.__file__}")
    print(f"[INIT] Python executável: {sys.executable}")
    print(f"[INIT] torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    # Verificar se é versão CPU-only
    if '+cpu' in torch.__version__:
        print(f"[INIT] ⚠️ AVISO: PyTorch CPU-only detectado ({torch.__version__})")
        print(f"[INIT] ⚠️ Instale PyTorch com CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[INIT] ✅ CUDA detectado: {gpu_name}")
            print(f"[INIT] ✅ CUDA version: {cuda_version}")
            
            # Testar criação de tensor
            test_tensor = torch.zeros(1).cuda()
            print(f"[INIT] ✅ Tensor de teste criado na GPU: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[INIT] ⚠️ CUDA disponível mas erro ao usar: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[INIT] ❌ CUDA não disponível")
        print(f"[INIT] Verificando variáveis de ambiente...")
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')
        print(f"[INIT] CUDA_VISIBLE_DEVICES: {cuda_visible}")
        print(f"[INIT] PATH contém CUDA: {'cuda' in os.environ.get('PATH', '').lower()}")
        
except Exception as e:
    print(f"[INIT] ❌ Erro ao importar/verificar torch: {e}")
    import traceback
    traceback.print_exc()
    # Definir torch como None para evitar erros posteriores
    torch = None

# Adicionar diretório src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importações com tratamento de erro mais claro
try:
    from image_generator import ImageGenerator
    from video_generator import VideoGenerator
except ImportError as e:
    st.error(f"❌ Erro ao importar módulos: {str(e)}")
    st.error("⚠️ Verifique se todas as dependências estão instaladas corretamente.")
    st.error("💡 Execute: pip install -r requirements.txt")
    st.stop()


# Configuração da página
st.set_page_config(
    page_title="Personagem Generativo e Animação",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Inicializa variáveis de sessão"""
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'generation_params' not in st.session_state:
        st.session_state.generation_params = {}
    if 'video_params' not in st.session_state:
        st.session_state.video_params = {}
    # Inicializar variáveis de vídeo
    if 'use_svd' not in st.session_state:
        st.session_state.use_svd = False
    if 'fps' not in st.session_state:
        st.session_state.fps = 3
    if 'duration_per_image' not in st.session_state:
        st.session_state.duration_per_image = 1.5
    if 'transition_frames' not in st.session_state:
        st.session_state.transition_frames = 15
    if 'add_loop' not in st.session_state:
        st.session_state.add_loop = True
    if 'svd_frames' not in st.session_state:
        st.session_state.svd_frames = 20  # Padrão: 20 frames
    if 'svd_fps' not in st.session_state:
        st.session_state.svd_fps = 4  # Padrão: 4 fps = 5 segundos com 20 frames


def main():
    """Função principal da aplicação"""
    init_session_state()

    # Título
    st.title("🎨 Personagem Generativo e Animação Curta")
    st.markdown("""
    ### Pipeline de IA Generativa para Criação de Personagens Animados
    Este sistema cria personagens visuais consistentes e gera uma animação curta em vídeo.

    ---
    **Instituição**: Universidade de Pernambuco (UPE)
    **Programa**: Residência em IA Generativa
    **Disciplina**: IA Generativa para Mídia Visual
    **Autores**: Vanthuir Maia e Rodrigo Santana
    """)

    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        # Verificar dispositivo com verificação robusta
        # torch já foi importado no topo do arquivo
        
        # Forçar detecção de CUDA
        has_cuda = False
        cuda_error = None
        
        try:
            # Verificar se CUDA está disponível
            if torch.cuda.is_available():
                # Tentar criar tensor na GPU para confirmar
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                has_cuda = True
                gpu_name = torch.cuda.get_device_name(0)
                device_info = f"🟢 GPU CUDA ({gpu_name})"
            else:
                device_info = "🔴 CPU (Lento)"
        except Exception as e:
            cuda_error = str(e)
            device_info = "🔴 CPU (Lento)"
            has_cuda = False

        st.info(f"**Dispositivo**: {device_info}")
        if has_cuda:
            st.success(f"✅ GPU detectada e funcionando!")
        else:
            st.warning("⚠️ Rodando em CPU. Geração será MUITO lenta (~5-10 min por imagem). Veja OTIMIZACOES_CPU.md")
            if cuda_error:
                with st.expander("🔍 Detalhes do erro CUDA"):
                    st.code(cuda_error)

        st.subheader("Geração de Imagens")

        # Preset de configurações
        preset = st.selectbox(
            "Preset de Velocidade",
            ["Ultra Rápido (CPU)", "Rápido", "Balanceado", "Alta Qualidade"],
            index=0 if not has_cuda else 2,
            help="Configurações pré-definidas. Ultra Rápido recomendado para CPU"
        )

        # Definir valores baseado no preset
        preset_configs = {
            "Ultra Rápido (CPU)": {"images": 3, "steps": 20, "guidance": 7.0},
            "Rápido": {"images": 5, "steps": 30, "guidance": 7.0},
            "Balanceado": {"images": 10, "steps": 50, "guidance": 7.5},
            "Alta Qualidade": {"images": 10, "steps": 80, "guidance": 8.0}
        }

        preset_config = preset_configs[preset]

        model_choice = st.selectbox(
            "Modelo",
            ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"],
            help="Escolha o modelo de geração de imagens"
        )

        num_images = st.slider(
            "Número de Imagens",
            min_value=1,
            max_value=20,
            value=preset_config["images"],
            help="Quantidade de imagens a gerar. CPU: recomendado 1-3 para teste"
        )

        seed = st.number_input(
            "Seed (0 = aleatório)",
            min_value=0,
            max_value=2**32-1,
            value=42,
            help="Seed para reprodutibilidade. Use 0 para seed aleatório"
        )

        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=preset_config["guidance"],
            step=0.5,
            help="Força de aderência ao prompt (7-15 recomendado)"
        )

        num_inference_steps = st.slider(
            "Passos de Inferência",
            min_value=10,
            max_value=100,
            value=preset_config["steps"],
            help="Mais passos = melhor qualidade (mas mais lento). CPU: use 20"
        )

        # Estimativa de tempo
        time_per_image_cpu = num_inference_steps * 0.15  # ~9 seg por step em CPU média
        time_per_image_gpu = num_inference_steps * 0.02  # ~1 seg por step em GPU média

        if has_cuda:
            estimated_time = (time_per_image_gpu * num_images) / 60
            st.info(f"⏱️ Tempo estimado: ~{estimated_time:.1f} minutos")
        else:
            estimated_time = (time_per_image_cpu * num_images) / 60
            st.warning(f"⏱️ Tempo estimado: ~{estimated_time:.0f} minutos")
            if estimated_time > 30:
                st.error(f"🚨 Isso vai demorar MUITO! Reduza imagens ou steps.")

        st.divider()

        st.subheader("Geração de Vídeo")
        
        # Seleção de método de animação
        animation_method = st.selectbox(
            "Método de Animação",
            ["Transições (OpenCV)", "IA - Stable Video Diffusion"],
            help="Transições: combina múltiplas imagens. SVD: anima imagem individual com IA (requer GPU)"
        )
        
        use_svd = animation_method == "IA - Stable Video Diffusion"
        
        # Salvar no session_state para usar na tab 3
        st.session_state.use_svd = use_svd
        
        if use_svd:
            # Configurações específicas do SVD
            st.info("🎨 **Stable Video Diffusion**: Anima uma imagem individual com movimento realista (requer GPU)")
            
            svd_resolution = st.selectbox(
                "Resolução",
                ["512x320 (Recomendado 8GB)", "384x256 (Ultra-Econômico)", "640x384 (Avançado)"],
                index=0,
                help="Resolução menor = menos memória GPU"
            )
            
            # Converter seleção para tupla
            resolution_map = {
                "512x320 (Recomendado 8GB)": (512, 320),
                "384x256 (Ultra-Econômico)": (384, 256),
                "640x384 (Avançado)": (640, 384)
            }
            svd_res = resolution_map[svd_resolution]
            
            svd_frames = st.slider(
                "Frames do Vídeo",
                min_value=15,
                max_value=25,
                value=20,
                help="Mais frames = mais memória. 20-25 frames = 5-10s de vídeo"
            )
            
            svd_fps = st.slider(
                "FPS do Vídeo",
                min_value=3,
                max_value=7,
                value=4,
                help="Frames por segundo. Menor FPS = vídeo mais longo"
            )
            
            # Calcular e mostrar duração estimada
            video_duration = svd_frames / svd_fps
            duration_color = "🟢" if 5 <= video_duration <= 10 else "🟡"
            st.info(f"{duration_color} **Duração estimada**: ~{video_duration:.1f} segundos ({svd_frames} frames ÷ {svd_fps} fps)")
            
            svd_steps = st.slider(
                "Passos de Inferência",
                min_value=20,
                max_value=30,
                value=25,
                help="Mais passos = melhor qualidade (mais lento)"
            )
            
            if has_cuda:
                estimated_svd_time = (svd_steps * 0.05 * svd_frames) / 60  # Estimativa
                st.info(f"⏱️ Tempo estimado: ~{estimated_svd_time:.1f} minutos")
            else:
                st.error("⚠️ SVD requer GPU CUDA. Use 'Transições (OpenCV)' em vez disso.")
            
            # Salvar parâmetros SVD no session_state
            st.session_state.svd_frames = svd_frames
            st.session_state.svd_fps = svd_fps
            st.session_state.svd_res = svd_res
            st.session_state.svd_steps = svd_steps
            
            # Inicializar variáveis OpenCV como None quando usando SVD
            fps = None
            duration_per_image = None
            transition_frames = None
            add_loop = None
                
        else:
            # Configurações tradicionais (OpenCV)
            fps = st.slider(
                "FPS (Frames por Segundo)",
                min_value=2,
                max_value=30,
                value=3,
                help="Velocidade do vídeo"
            )

            duration_per_image = st.slider(
                "Duração por Imagem (s)",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Quanto tempo cada imagem aparece"
            )

            transition_frames = st.slider(
                "Frames de Transição",
                min_value=5,
                max_value=30,
                value=15,
                help="Suavidade da transição entre imagens"
            )

            add_loop = st.checkbox(
                "Adicionar Loop",
                value=True,
                help="Criar transição de volta para primeira imagem"
            )
            
            # Salvar parâmetros OpenCV no session_state
            st.session_state.fps = fps
            st.session_state.duration_per_image = duration_per_image
            st.session_state.transition_frames = transition_frames
            st.session_state.add_loop = add_loop
            
            # Inicializar variáveis SVD como None quando usando OpenCV
            svd_frames = None
            svd_fps = None
            svd_res = None
            svd_steps = None

    # Área principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Geração",
        "🖼️ Imagens",
        "🎬 Vídeo",
        "📊 Documentação"
    ])

    # Tab 1: Geração
    with tab1:
        st.header("Criação do Personagem")

        # Prompt do personagem
        st.subheader("Descrição do Personagem")
        prompt = st.text_area(
            "Descreva seu personagem em detalhes",
            value=(
                "A cute cartoon robot character, round body, big expressive eyes, "
                "friendly smile, blue and white colors, simple design, "
                "mascot style, standing pose, white background, "
                "digital art, high quality, consistent character design"
            ),
            height=150,
            help="Seja específico sobre aparência, estilo, cores, pose, etc."
        )

        negative_prompt = st.text_area(
            "Prompt Negativo (opcional)",
            value=(
                "blurry, low quality, distorted, deformed, ugly, "
                "bad anatomy, bad proportions, extra limbs, "
                "text, watermark, signature"
            ),
            height=100,
            help="O que você quer evitar nas imagens"
        )

        # Botão de geração
        col1, col2 = st.columns([1, 3])
        with col1:
            generate_btn = st.button(
                "🎨 Gerar Imagens",
                type="primary",
                width='stretch'
            )

        if generate_btn:
            if not prompt.strip():
                st.error("Por favor, forneça uma descrição do personagem!")
            else:
                # Placeholder para progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                timer_container = st.empty()

                # Iniciar contador de tempo
                start_time = time.time()
                elapsed_time = 0

                try:
                    # Criar gerador
                    status_text.text("🔧 Carregando modelo...")
                    load_start = time.time()
                    # Forçar uso de CUDA se disponível com verificação robusta
                    # torch já foi importado no topo do arquivo
                    device_param = "auto"  # Deixar ImageGenerator detectar automaticamente
                    try:
                        # Verificar CUDA de forma robusta (se torch estiver disponível)
                        if torch is not None and torch.cuda.is_available():
                            # Testar se CUDA realmente funciona
                            test = torch.zeros(1).cuda()
                            del test
                            torch.cuda.empty_cache()
                            device_param = "cuda"
                            status_text.text(f"🔧 Carregando modelo na GPU...")
                        else:
                            device_param = "cpu"
                            status_text.text(f"🔧 Carregando modelo na CPU...")
                    except Exception as e:
                        device_param = "cpu"
                        status_text.text(f"🔧 CUDA não disponível, usando CPU...")
                    
                    generator = ImageGenerator(model_id=model_choice, device=device_param)
                    load_time = time.time() - load_start
                    
                    timer_container.info(f"⏱️ **Tempo de carregamento do modelo**: {load_time:.1f} segundos")

                    # Criar diretório único para esta geração
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = f"outputs/images/{timestamp}"

                    # Função de callback para atualizar progresso
                    def update_progress(current, total, status, time_remaining=0, time_per_image=0):
                        nonlocal elapsed_time
                        
                        progress = current / total
                        progress_bar.progress(progress)
                        
                        # Calcular tempo decorrido total
                        elapsed_time = time.time() - start_time
                        elapsed_mins = int(elapsed_time // 60)
                        elapsed_secs = int(elapsed_time % 60)

                        if status == "generating":
                            status_text.text(f"🎨 Gerando imagem {current + 1}/{total}...")
                            
                            # Mostrar tempo decorrido
                            timer_container.markdown(
                                f"""
                                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
                                    <h4 style='margin: 0; color: #1f77b4;'>⏱️ Contador de Tempo</h4>
                                    <p style='margin: 5px 0;'><strong>Tempo decorrido:</strong> {elapsed_mins}min {elapsed_secs}s</p>
                                    <p style='margin: 5px 0;'><strong>Imagem atual:</strong> {current + 1}/{total}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        elif status == "completed":
                            status_text.text(f"✅ Imagem {current}/{total} concluída!")
                            
                            if time_remaining > 0:
                                mins_remaining = int(time_remaining // 60)
                                secs_remaining = int(time_remaining % 60)
                                
                                # Calcular tempo total estimado
                                total_estimated = elapsed_time + time_remaining
                                total_mins = int(total_estimated // 60)
                                total_secs = int(total_estimated % 60)

                                time_text.markdown(
                                    f"⏱️ **Tempo médio por imagem**: {time_per_image:.1f}s"
                                )
                                
                                timer_container.markdown(
                                    f"""
                                    <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;'>
                                        <h4 style='margin: 0; color: #2e7d32;'>⏱️ Contador de Tempo</h4>
                                        <p style='margin: 5px 0;'><strong>⏳ Tempo decorrido:</strong> {elapsed_mins}min {elapsed_secs}s</p>
                                        <p style='margin: 5px 0;'><strong>📊 Tempo por imagem:</strong> {time_per_image:.1f}s</p>
                                        <p style='margin: 5px 0;'><strong>⏰ Tempo restante estimado:</strong> {mins_remaining}min {secs_remaining}s</p>
                                        <p style='margin: 5px 0;'><strong>🎯 Tempo total estimado:</strong> {total_mins}min {total_secs}s</p>
                                        <p style='margin: 5px 0;'><strong>📈 Progresso:</strong> {current}/{total} imagens ({progress*100:.1f}%)</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                # Finalização
                                total_mins = int(elapsed_time // 60)
                                total_secs = int(elapsed_time % 60)
                                avg_time = elapsed_time / total if total > 0 else 0
                                
                                timer_container.markdown(
                                    f"""
                                    <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;'>
                                        <h4 style='margin: 0; color: #e65100;'>🎉 Processamento Concluído!</h4>
                                        <p style='margin: 5px 0;'><strong>⏱️ Tempo total:</strong> {total_mins}min {total_secs}s</p>
                                        <p style='margin: 5px 0;'><strong>📊 Tempo médio por imagem:</strong> {avg_time:.1f}s</p>
                                        <p style='margin: 5px 0;'><strong>✅ Total de imagens:</strong> {total}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    # Gerar imagens com callback
                    status_text.text(f"🚀 Iniciando geração de {num_images} imagens...")

                    images = generator.generate_images(
                        prompt=prompt,
                        num_images=num_images,
                        negative_prompt=negative_prompt if negative_prompt.strip() else None,
                        seed=seed if seed > 0 else None,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        output_dir=output_dir,
                        progress_callback=update_progress
                    )

                    # Salvar na sessão
                    st.session_state.generated_images = images
                    st.session_state.generation_params = generator.get_generation_params()
                    st.session_state.output_dir = output_dir

                    # Limpar memória
                    generator.cleanup()

                    # Calcular tempo total final
                    total_time = time.time() - start_time
                    total_mins = int(total_time // 60)
                    total_secs = int(total_time % 60)
                    avg_time = total_time / num_images if num_images > 0 else 0

                    # Finalizar
                    progress_bar.progress(1.0)
                    status_text.empty()
                    time_text.empty()

                    # Mostrar resumo final
                    st.success(f"✅ {len(images)} imagens geradas com sucesso!")
                    st.info(f"📁 Imagens salvas em: {output_dir}")
                    
                    # Exibir resumo de tempo final
                    timer_container.markdown(
                        f"""
                        <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin-top: 20px;'>
                            <h3 style='margin: 0 0 15px 0; color: #1976d2;'>📊 Resumo do Processamento</h3>
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                                <div>
                                    <p style='margin: 5px 0; font-size: 16px;'><strong>⏱️ Tempo Total:</strong></p>
                                    <p style='margin: 0; font-size: 24px; color: #1976d2; font-weight: bold;'>{total_mins}min {total_secs}s</p>
                                </div>
                                <div>
                                    <p style='margin: 5px 0; font-size: 16px;'><strong>📊 Tempo Médio/Imagem:</strong></p>
                                    <p style='margin: 0; font-size: 24px; color: #1976d2; font-weight: bold;'>{avg_time:.1f}s</p>
                                </div>
                                <div>
                                    <p style='margin: 5px 0; font-size: 16px;'><strong>🔧 Tempo de Carregamento:</strong></p>
                                    <p style='margin: 0; font-size: 20px; color: #1976d2;'>{load_time:.1f}s</p>
                                </div>
                                <div>
                                    <p style='margin: 5px 0; font-size: 16px;'><strong>🎨 Tempo de Geração:</strong></p>
                                    <p style='margin: 0; font-size: 20px; color: #1976d2;'>{total_time - load_time:.1f}s</p>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.balloons()

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    time_text.empty()
                    timer_container.empty()
                    
                    # Calcular tempo até o erro
                    error_time = time.time() - start_time
                    error_mins = int(error_time // 60)
                    error_secs = int(error_time % 60)
                    
                    st.error(f"❌ Erro ao gerar imagens: {str(e)}")
                    st.warning(f"⏱️ Processamento interrompido após {error_mins}min {error_secs}s")
                    st.exception(e)

    # Tab 2: Visualização de Imagens
    with tab2:
        st.header("Imagens Geradas")

        if st.session_state.generated_images:
            st.success(f"Total de imagens: {len(st.session_state.generated_images)}")

            # Exibir parâmetros de geração
            with st.expander("📋 Parâmetros de Geração"):
                params = st.session_state.generation_params
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Prompt:** {params.get('prompt', 'N/A')}")
                    st.write(f"**Seed:** {params.get('seed', 'N/A')}")
                    st.write(f"**Guidance Scale:** {params.get('guidance_scale', 'N/A')}")
                with col2:
                    st.write(f"**Passos:** {params.get('num_inference_steps', 'N/A')}")
                    st.write(f"**Resolução:** {params.get('width', 'N/A')}x{params.get('height', 'N/A')}")
                    st.write(f"**Modelo:** {params.get('model_id', 'N/A')}")

            # Mostrar imagens em grade
            st.subheader("Galeria de Imagens")
            cols = st.columns(3)
            for idx, img in enumerate(st.session_state.generated_images):
                with cols[idx % 3]:
                    st.image(img, caption=f"Imagem {idx + 1}", width='stretch')

        else:
            st.info("Nenhuma imagem gerada ainda. Vá para a aba 'Geração' para criar seu personagem!")

    # Tab 3: Geração de Vídeo
    with tab3:
        st.header("Geração de Vídeo")

        if st.session_state.generated_images:
            st.success(f"Pronto para criar vídeo com {len(st.session_state.generated_images)} imagens")

            # Verificar se está usando SVD (anima uma imagem por vez)
            use_svd = st.session_state.get('use_svd', False)
            
            # Seletor de imagem para SVD
            selected_image_idx = 0
            if use_svd and len(st.session_state.generated_images) > 1:
                st.subheader("📸 Escolher Imagem para Animar")
                st.info("💡 SVD anima uma imagem por vez. Escolha qual imagem deseja animar:")
                
                # Criar colunas para mostrar miniaturas
                num_cols = min(4, len(st.session_state.generated_images))
                cols = st.columns(num_cols)
                
                # Criar opções para o selectbox
                image_options = []
                for i, img in enumerate(st.session_state.generated_images):
                    image_options.append(f"Imagem {i+1}")
                
                # Selectbox para escolher a imagem
                selected_option = st.selectbox(
                    "Selecione a imagem:",
                    options=image_options,
                    index=0,
                    help="Escolha qual imagem será animada pelo Stable Video Diffusion"
                )
                
                # Extrair índice da opção selecionada
                selected_image_idx = image_options.index(selected_option)
                
                # Mostrar preview da imagem selecionada
                st.image(
                    st.session_state.generated_images[selected_image_idx],
                    caption=f"Imagem selecionada: {selected_option}",
                    width=300
                )
            elif use_svd:
                # Se só tem uma imagem, usar ela
                selected_image_idx = 0
                st.info("💡 SVD anima uma imagem por vez. Usando a única imagem disponível.")
                st.image(
                    st.session_state.generated_images[0],
                    caption="Imagem que será animada",
                    width=300
                )

            # Botão de geração de vídeo
            col1, col2 = st.columns([1, 3])
            with col1:
                generate_video_btn = st.button(
                    "🎬 Gerar Vídeo",
                    type="primary",
                    width='stretch'
                )

            if generate_video_btn:
                # Recuperar método do session_state
                use_svd = st.session_state.get('use_svd', False)
                
                if use_svd:
                    # Método SVD - Animar imagem individual
                    if not has_cuda:
                        st.error("❌ Stable Video Diffusion requer GPU CUDA. Selecione 'Transições (OpenCV)' ou instale PyTorch com CUDA.")
                    else:
                        with st.spinner("🎨 Gerando vídeo com IA (Stable Video Diffusion)... Isso pode levar algumas minutos/horas."):
                            try:
                                # Recuperar parâmetros do session_state
                                svd_frames = st.session_state.get('svd_frames', 20)
                                svd_fps = st.session_state.get('svd_fps', 4)
                                svd_res = st.session_state.get('svd_res', (512, 320))
                                svd_steps = st.session_state.get('svd_steps', 25)
                                
                                # Criar gerador de vídeo
                                video_gen = VideoGenerator()

                                # Criar diretório de vídeo
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                video_output = f"outputs/videos/svd_animation_{timestamp}.mp4"

                                # Usar imagem selecionada (ou primeira se não foi selecionada)
                                image_to_animate = st.session_state.generated_images[selected_image_idx]
                                
                                st.info(f"🎨 Animando Imagem {selected_image_idx + 1} de {len(st.session_state.generated_images)} disponíveis.")

                                # Gerar vídeo com SVD
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Callback para atualizar progresso
                                def update_progress(progress, status_msg):
                                    progress_bar.progress(progress)
                                    status_text.text(status_msg)
                                
                                # Inicializar com progresso de download
                                status_text.text("🔧 Preparando modelo SVD...")
                                progress_bar.progress(0.05)
                                
                                video_path = video_gen.animate_image_svd(
                                    image=image_to_animate,
                                    output_path=video_output,
                                    num_frames=svd_frames,
                                    fps=svd_fps,
                                    resolution=svd_res,
                                    num_inference_steps=svd_steps,
                                    decode_chunk_size=1,
                                    progress_callback=update_progress
                                )
                                
                                progress_bar.progress(1.0)
                                status_text.text("✅ Vídeo gerado com sucesso!")

                                # Salvar na sessão
                                st.session_state.video_path = video_path
                                st.session_state.video_params = video_gen.get_video_params()

                                st.success("✅ Vídeo gerado com sucesso usando Stable Video Diffusion!")
                                
                                # Limpar memória
                                video_gen.cleanup_svd()

                            except RuntimeError as e:
                                error_msg = str(e)
                                if "Out of Memory" in error_msg or "OOM" in error_msg:
                                    st.error("❌ Memória GPU insuficiente!")
                                    st.warning("💡 Tente:")
                                    st.write("- Reduzir resolução para '384x256'")
                                    st.write("- Reduzir frames para 10")
                                    st.write("- Fechar outros programas usando GPU")
                                    st.write("- Ou use 'Transições (OpenCV)' que não requer GPU")
                                else:
                                    st.error(f"Erro ao gerar vídeo: {error_msg}")
                                    st.exception(e)
                            except Exception as e:
                                st.error(f"Erro ao gerar vídeo: {str(e)}")
                                st.exception(e)
                            finally:
                                progress_bar.empty()
                                status_text.empty()
                else:
                    # Método tradicional (OpenCV)
                    # Recuperar parâmetros do session_state
                    fps = st.session_state.get('fps', 3)
                    duration_per_image = st.session_state.get('duration_per_image', 1.5)
                    transition_frames = st.session_state.get('transition_frames', 15)
                    add_loop = st.session_state.get('add_loop', True)
                    
                    if fps is None:
                        st.error("⚠️ Erro: Configurações de vídeo não encontradas. Configure na sidebar primeiro.")
                    else:
                        with st.spinner("Gerando vídeo... Aguarde."):
                            try:
                                # Criar gerador de vídeo
                                video_gen = VideoGenerator()

                                # Criar diretório de vídeo
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                video_output = f"outputs/videos/animation_{timestamp}.mp4"

                                # Gerar vídeo
                                video_path = video_gen.create_video_from_images(
                                    images=st.session_state.generated_images,
                                    output_path=video_output,
                                    fps=fps,
                                    duration_per_image=duration_per_image,
                                    transition_frames=transition_frames,
                                    add_loop=add_loop
                                )

                                # Salvar na sessão
                                st.session_state.video_path = video_path
                                st.session_state.video_params = video_gen.get_video_params()

                                st.success("✅ Vídeo gerado com sucesso!")

                            except Exception as e:
                                st.error(f"Erro ao gerar vídeo: {str(e)}")
                                st.exception(e)

            # Mostrar vídeo se disponível
            if st.session_state.video_path:
                st.subheader("Vídeo Gerado")

                # Exibir parâmetros do vídeo
                with st.expander("📋 Parâmetros do Vídeo"):
                    vparams = st.session_state.video_params
                    method = vparams.get('method', 'opencv')
                    
                    if method == 'stable_video_diffusion':
                        # Parâmetros SVD
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Método:** 🎨 Stable Video Diffusion (IA)")
                            st.write(f"**Frames:** {vparams.get('num_frames', 'N/A')}")
                            st.write(f"**FPS:** {vparams.get('fps', 'N/A')}")
                            st.write(f"**Resolução:** {vparams.get('resolution', 'N/A')}")
                        with col2:
                            st.write(f"**Resolução Original:** {vparams.get('original_resolution', 'N/A')}")
                            st.write(f"**Passos:** {vparams.get('num_inference_steps', 'N/A')}")
                            st.write(f"**Duração:** ~{vparams.get('duration', 'N/A'):.2f}s")
                            st.write(f"**Memória GPU:** {vparams.get('gpu_memory_used', 'N/A')}")
                    else:
                        # Parâmetros OpenCV tradicional
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Método:** 🎬 Transições (OpenCV)")
                            st.write(f"**Número de Imagens:** {vparams.get('num_images', 'N/A')}")
                            st.write(f"**FPS:** {vparams.get('fps', 'N/A')}")
                            st.write(f"**Duração por Imagem:** {vparams.get('duration_per_image', 'N/A')}s")
                        with col2:
                            st.write(f"**Frames de Transição:** {vparams.get('transition_frames', 'N/A')}")
                            st.write(f"**Duração Total:** {vparams.get('total_duration', 'N/A'):.2f}s")
                            st.write(f"**Resolução:** {vparams.get('resolution', 'N/A')}")
                            st.write(f"**Loop:** {'Sim' if vparams.get('add_loop', False) else 'Não'}")

                # Reproduzir vídeo
                st.video(st.session_state.video_path)

                # Botão de download
                with open(st.session_state.video_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download do Vídeo",
                        data=file,
                        file_name=Path(st.session_state.video_path).name,
                        mime="video/mp4"
                    )

        else:
            st.info("Gere as imagens primeiro antes de criar o vídeo!")

    # Tab 4: Documentação
    with tab4:
        st.header("Documentação Técnica")

        st.markdown("""
        ## Pipeline de Geração

        ### 1. Geração de Imagens
        - **Modelo**: Stable Diffusion (Hugging Face Diffusers)
        - **Técnica**: Text-to-Image com controle de seed
        - **Estratégia de Consistência**: Seeds sequenciais a partir de uma seed base
        - **Parâmetros Principais**:
          - Guidance Scale: controla aderência ao prompt
          - Inference Steps: qualidade da geração
          - Negative Prompt: evita características indesejadas

        ### 2. Geração de Vídeo
        - **Técnica**: Interpolação linear entre frames (cross-dissolve)
        - **Biblioteca**: OpenCV
        - **Processo**:
          1. Cada imagem é mantida por N frames (definido por FPS × duração)
          2. Transições suaves entre imagens usando cv2.addWeighted
          3. Loop opcional para criar animação contínua

        ### 3. Ferramentas Utilizadas
        - **diffusers**: Geração de imagens com Stable Diffusion
        - **transformers**: Modelos de linguagem para processamento de prompts
        - **torch**: Backend de deep learning
        - **opencv-python**: Processamento de vídeo
        - **streamlit**: Interface web interativa

        ### 4. Desafios e Soluções

        #### Consistência Visual
        - **Desafio**: Manter identidade do personagem entre imagens
        - **Solução**: Uso de seeds sequenciais e prompt detalhado

        #### Coerência Temporal
        - **Desafio**: Transições suaves no vídeo
        - **Solução**: Interpolação linear entre frames

        #### Limitações
        - Modelos locais requerem GPU com boa memória
        - Geração pode ser lenta em hardware limitado
        - Consistência não é perfeita (variações podem ocorrer)

        ### 5. Melhorias Futuras
        - Implementar ControlNet para maior controle
        - Adicionar motion transfer com MediaPipe
        - Integrar modelos text-to-video (Gen-2, Pika)
        - Adicionar efeitos de zoom, pan, rotate
        """)

        # Exportar documentação completa
        if st.session_state.generation_params and st.session_state.video_params:
            st.subheader("Exportar Documentação Completa")

            doc_data = {
                "projeto": "Personagem Generativo e Animação Curta",
                "timestamp": datetime.now().isoformat(),
                "geração_imagens": st.session_state.generation_params,
                "geração_vídeo": st.session_state.video_params,
                "pipeline": {
                    "etapa_1": "Geração de imagens com Stable Diffusion",
                    "etapa_2": "Criação de vídeo com interpolação de frames",
                    "ferramentas": ["Stable Diffusion", "OpenCV", "Streamlit"],
                }
            }

            doc_json = json.dumps(doc_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="📄 Download Documentação (JSON)",
                data=doc_json,
                file_name=f"documentacao_tecnica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
