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

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_generator import ImageGenerator
from video_generator import VideoGenerator


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

        # Verificar dispositivo
        import torch
        has_cuda = torch.cuda.is_available()
        device_info = "🟢 GPU CUDA" if has_cuda else "🔴 CPU (Lento)"

        st.info(f"**Dispositivo**: {device_info}")
        if not has_cuda:
            st.warning("⚠️ Rodando em CPU. Geração será MUITO lenta (~5-10 min por imagem). Veja OTIMIZACOES_CPU.md")

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
                use_container_width=True
            )

        if generate_btn:
            if not prompt.strip():
                st.error("Por favor, forneça uma descrição do personagem!")
            else:
                # Placeholder para progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()

                try:
                    # Criar gerador
                    status_text.text("🔧 Carregando modelo...")
                    generator = ImageGenerator(model_id=model_choice)

                    # Criar diretório único para esta geração
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = f"outputs/images/{timestamp}"

                    # Função de callback para atualizar progresso
                    def update_progress(current, total, status, time_remaining=0, time_per_image=0):
                        progress = current / total
                        progress_bar.progress(progress)

                        if status == "generating":
                            status_text.text(f"🎨 Gerando imagem {current + 1}/{total}...")
                        elif status == "completed":
                            mins_remaining = int(time_remaining // 60)
                            secs_remaining = int(time_remaining % 60)

                            status_text.text(f"✅ Imagem {current}/{total} concluída!")

                            if time_remaining > 0:
                                time_text.markdown(
                                    f"⏱️ **Tempo médio por imagem**: {time_per_image:.1f}s | "
                                    f"**Tempo restante estimado**: {mins_remaining}min {secs_remaining}s"
                                )
                            else:
                                time_text.markdown(f"🎉 **Todas as imagens geradas!**")

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

                    # Finalizar
                    progress_bar.progress(1.0)
                    status_text.empty()
                    time_text.empty()

                    st.success(f"✅ {len(images)} imagens geradas com sucesso!")
                    st.info(f"📁 Imagens salvas em: {output_dir}")
                    st.balloons()

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    time_text.empty()
                    st.error(f"❌ Erro ao gerar imagens: {str(e)}")
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
                    st.image(img, caption=f"Imagem {idx + 1}", use_container_width=True)

        else:
            st.info("Nenhuma imagem gerada ainda. Vá para a aba 'Geração' para criar seu personagem!")

    # Tab 3: Geração de Vídeo
    with tab3:
        st.header("Geração de Vídeo")

        if st.session_state.generated_images:
            st.success(f"Pronto para criar vídeo com {len(st.session_state.generated_images)} imagens")

            # Botão de geração de vídeo
            col1, col2 = st.columns([1, 3])
            with col1:
                generate_video_btn = st.button(
                    "🎬 Gerar Vídeo",
                    type="primary",
                    use_container_width=True
                )

            if generate_video_btn:
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Número de Imagens:** {vparams.get('num_images', 'N/A')}")
                        st.write(f"**FPS:** {vparams.get('fps', 'N/A')}")
                        st.write(f"**Duração por Imagem:** {vparams.get('duration_per_image', 'N/A')}s")
                    with col2:
                        st.write(f"**Frames de Transição:** {vparams.get('transition_frames', 'N/A')}")
                        st.write(f"**Duração Total:** {vparams.get('total_duration', 'N/A'):.2f}s")
                        st.write(f"**Resolução:** {vparams.get('resolution', 'N/A')}")

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
