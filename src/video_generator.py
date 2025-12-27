"""
Módulo de Geração de Vídeo
Pipeline para criar vídeos animados a partir de imagens de personagens

Instituição: Universidade de Pernambuco (UPE)
Programa: Residência em IA Generativa
Disciplina: IA Generativa para Mídia Visual
Autores: Vanthuir Maia e Rodrigo Santana
"""

import os
from typing import List, Optional, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
import torch
import warnings
# Suprimir avisos de autocast quando CUDA não está disponível (vem do diffusers)
warnings.filterwarnings('ignore', message='.*User provided device_type of.*')


class VideoGenerator:
    """
    Gerador de vídeos animados a partir de imagens com transições suaves
    """

    def __init__(self):
        """Inicializa o gerador de vídeo"""
        self.video_params = {}
        self.svd_pipeline = None

    def create_video_from_images(
        self,
        images: Union[List[Image.Image], List[str]],
        output_path: str = "outputs/videos/character_animation.mp4",
        fps: int = 2,
        duration_per_image: float = 1.0,
        transition_frames: int = 10,
        add_loop: bool = True,
        video_codec: str = "mp4v",
        save_metadata: bool = True
    ) -> str:
        """
        Cria um vídeo a partir de uma lista de imagens

        Args:
            images: Lista de imagens PIL ou caminhos de arquivos
            output_path: Caminho do arquivo de saída
            fps: Frames por segundo (2-10 recomendado para animação)
            duration_per_image: Duração de cada imagem em segundos
            transition_frames: Número de frames de transição entre imagens
            add_loop: Adicionar transição de volta para primeira imagem
            video_codec: Codec de vídeo ('mp4v', 'avc1', 'XVID')
            save_metadata: Salvar metadados da geração

        Returns:
            Caminho do vídeo gerado
        """
        # Criar diretório de saída
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Carregar imagens se forem caminhos
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img))
            else:
                loaded_images.append(img)

        if len(loaded_images) == 0:
            raise ValueError("Nenhuma imagem fornecida")

        print(f"Criando vídeo com {len(loaded_images)} imagens...")
        print(f"FPS: {fps}, Duração por imagem: {duration_per_image}s")

        # Calcular número de frames para cada imagem
        frames_per_image = int(fps * duration_per_image)

        # Obter dimensões da primeira imagem
        first_img = loaded_images[0]
        width, height = first_img.size

        # Configurar writer de vídeo
        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        video_writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            fps,
            (width, height)
        )

        total_frames = 0

        # Processar cada imagem
        for i, img in enumerate(loaded_images):
            # Converter PIL para OpenCV (BGR)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Adicionar frames estáticos da imagem
            for _ in range(frames_per_image):
                video_writer.write(img_cv)
                total_frames += 1

            # Adicionar transição para próxima imagem
            if i < len(loaded_images) - 1:
                next_img = loaded_images[i + 1]
                next_img_cv = cv2.cvtColor(np.array(next_img), cv2.COLOR_RGB2BGR)

                transition_frames_generated = self._create_transition(
                    img_cv, next_img_cv, transition_frames, video_writer
                )
                total_frames += transition_frames_generated

                print(f"  Imagem {i+1} -> {i+2}: {frames_per_image} frames + {transition_frames} transição")

        # Adicionar loop (transição de volta para primeira imagem)
        if add_loop and len(loaded_images) > 1:
            last_img_cv = cv2.cvtColor(np.array(loaded_images[-1]), cv2.COLOR_RGB2BGR)
            first_img_cv = cv2.cvtColor(np.array(loaded_images[0]), cv2.COLOR_RGB2BGR)

            transition_frames_generated = self._create_transition(
                last_img_cv, first_img_cv, transition_frames, video_writer
            )
            total_frames += transition_frames_generated
            print(f"  Loop: {transition_frames} frames de transição")

        # Finalizar vídeo
        video_writer.release()

        # Calcular duração total
        total_duration = total_frames / fps

        print(f"\nVídeo criado com sucesso!")
        print(f"  Total de frames: {total_frames}")
        print(f"  Duração: {total_duration:.2f} segundos")
        print(f"  Salvo em: {output_file}")

        # Salvar metadados
        if save_metadata:
            self.video_params = {
                "num_images": len(loaded_images),
                "fps": fps,
                "duration_per_image": duration_per_image,
                "transition_frames": transition_frames,
                "total_frames": total_frames,
                "total_duration": total_duration,
                "resolution": f"{width}x{height}",
                "codec": video_codec,
                "add_loop": add_loop,
                "timestamp": datetime.now().isoformat()
            }

            metadata_path = output_file.parent / "video_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.video_params, f, indent=2, ensure_ascii=False)
            print(f"  Metadados salvos em: {metadata_path}")

        return str(output_file)

    def _create_transition(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        num_frames: int,
        video_writer: cv2.VideoWriter
    ) -> int:
        """
        Cria transição suave entre duas imagens usando interpolação

        Args:
            img1: Imagem inicial (formato OpenCV)
            img2: Imagem final (formato OpenCV)
            num_frames: Número de frames de transição
            video_writer: Writer de vídeo

        Returns:
            Número de frames gerados
        """
        if num_frames <= 0:
            return 0

        for i in range(num_frames):
            # Calcular peso de interpolação (0.0 a 1.0)
            alpha = (i + 1) / (num_frames + 1)

            # Interpolar entre as duas imagens
            blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

            # Escrever frame
            video_writer.write(blended)

        return num_frames

    def create_video_with_effects(
        self,
        images: Union[List[Image.Image], List[str]],
        output_path: str = "outputs/videos/character_animation_effects.mp4",
        fps: int = 10,
        effect: str = "zoom",
        **kwargs
    ) -> str:
        """
        Cria vídeo com efeitos especiais (zoom, pan, rotate)

        Args:
            images: Lista de imagens
            output_path: Caminho de saída
            fps: Frames por segundo
            effect: Tipo de efeito ('zoom', 'pan', 'fade')
            **kwargs: Parâmetros adicionais

        Returns:
            Caminho do vídeo gerado
        """
        # Implementação futura: adicionar efeitos como zoom, pan, etc.
        print(f"Efeito '{effect}' será implementado em versão futura")
        return self.create_video_from_images(images, output_path, fps, **kwargs)

    def get_video_params(self) -> dict:
        """Retorna os parâmetros do último vídeo gerado"""
        return self.video_params.copy()
    
    def _init_svd_pipeline(self, progress_callback=None):
        """
        Inicializa pipeline Stable Video Diffusion otimizado para 8GB VRAM
        Usa todas as técnicas de economia de memória possíveis
        
        Args:
            progress_callback: Função callback(progress, status) para atualizar progresso
                              progress: float entre 0.0 e 1.0
                              status: str com mensagem de status
        """
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image, export_to_video
            from huggingface_hub import hf_hub_download
            import os
        except ImportError:
            raise ImportError(
                "Stable Video Diffusion requer diffusers>=0.30.0. "
                "Execute: pip install diffusers[torch]"
            )
        
        if self.svd_pipeline is None:
            if progress_callback:
                progress_callback(0.05, "🔧 Preparando download do modelo SVD...")
            else:
                print("\n" + "="*60)
                print("Carregando Stable Video Diffusion (otimizado para 8GB)...")
                print("="*60)
            
            # Verificar se tem GPU
            if not torch.cuda.is_available():
                raise RuntimeError("GPU CUDA não disponível. SVD requer GPU.")
            
            if progress_callback:
                progress_callback(0.1, "🔧 Configurando otimizações para 8GB VRAM...")
            else:
                # Configurações MÁXIMAS de economia de memória para RTX 3050 8GB
                print("🔧 Configurações de otimização de memória:")
                print("  - FP16 (metade da memória)")
                print("  - CPU Offloading (move partes para RAM)")
                print("  - Attention Slicing máximo")
                print("  - Resolução reduzida: 512x320")
                print("  - Frames otimizados: 20")
                print("="*60 + "\n")
            
            try:
                if progress_callback:
                    progress_callback(0.15, "📥 Baixando modelo SVD (~5GB). Isso pode levar alguns minutos...")
                    progress_callback(0.2, "💡 Na primeira execução, o modelo será baixado automaticamente")
                    progress_callback(0.25, "📦 Baixando arquivos do modelo... (veja progresso no terminal)")
                
                # Carregar pipeline (vai baixar automaticamente se necessário)
                # O download será mostrado no terminal com barra de progresso do Hugging Face
                # Não podemos atualizar Streamlit de threads separadas, então mostramos mensagem estática
                if progress_callback:
                    progress_callback(0.3, "📥 Download em andamento... (veja progresso no terminal)")
                
                self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16,  # CRÍTICO: FP16 economiza 50% memória
                    variant="fp16"  # Usar variante FP16 pré-compilada
                )
                
                if progress_callback:
                    progress_callback(0.8, "✅ Modelo baixado! Carregando na memória...")
                
                # OTIMIZAÇÕES CRÍTICAS PARA 8GB:
                
                # 1. CPU Offloading - move partes do modelo para CPU RAM
                # Isso economiza MUITA memória GPU
                self.svd_pipeline.enable_model_cpu_offload()
                
                # 2. Attention Slicing - processa atenção em chunks
                # Reduz pico de memória durante inferência
                self.svd_pipeline.enable_attention_slicing(slice_size="max")
                
                # 3. VAE Slicing - Não disponível no SVD Pipeline
                # O SVD já otimiza o VAE internamente, não precisa de slicing manual
                # Nota: enable_vae_slicing() não existe no StableVideoDiffusionPipeline
                
                if progress_callback:
                    progress_callback(0.95, "⚙️ Aplicando otimizações...")
                
                if progress_callback:
                    progress_callback(1.0, "✅ Modelo SVD carregado e pronto!")
                else:
                    print("✅ Pipeline SVD carregado e otimizado com sucesso!")
                    print(f"💾 Memória GPU estimada: ~6-7 GB\n")
                
            except Exception as e:
                print(f"❌ Erro ao carregar SVD: {e}")
                raise
    
    def _check_gpu_memory(self) -> dict:
        """
        Verifica memória GPU disponível
        """
        if not torch.cuda.is_available():
            return {"available": False, "total": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        free = total - allocated
        
        return {
            "available": True,
            "total": total,
            "allocated": allocated,
            "free": free
        }
    
    def animate_image_svd(
        self,
        image: Union[Image.Image, str],
        output_path: str = "outputs/videos/svd_animation.mp4",
        num_frames: int = 20,  # Padrão: 20 frames para ~5s de vídeo
        fps: int = 4,  # Padrão: 4 fps para duração adequada
        resolution: tuple = (512, 320),  # Resolução reduzida para 8GB
        num_inference_steps: int = 25,  # Reduzido para ser mais rápido
        motion_bucket_id: int = 127,
        decode_chunk_size: int = 1,  # Processar 1 frame por vez
        save_metadata: bool = True,
        progress_callback=None  # Callback para progresso: callback(progress, status)
    ) -> str:
        """
        Anima uma única imagem usando Stable Video Diffusion
        OTIMIZADO PARA RTX 3050 8GB com máximo de economia de memória
        
        Args:
            image: Imagem PIL ou caminho de arquivo
            output_path: Caminho do vídeo de saída
            num_frames: Número de frames (20-25 recomendado para 5-10s de vídeo)
            fps: Frames por segundo do vídeo (3-5 fps recomendado para duração adequada)
            resolution: Resolução (width, height) - usar (512, 320) para 8GB
            num_inference_steps: Passos de inferência (25 é bom equilíbrio)
            motion_bucket_id: Controla quantidade de movimento (127 = médio)
            decode_chunk_size: Frames a processar por vez (1 = mínimo memória)
            save_metadata: Salvar metadados em JSON
            
        Returns:
            Caminho do vídeo gerado
            
        Raises:
            RuntimeError: Se GPU não disponível ou memória insuficiente
        """
        try:
            from diffusers.utils import load_image, export_to_video
        except ImportError:
            raise ImportError("diffusers não instalado. Execute: pip install diffusers[torch]")
        
        # Verificar GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA não disponível")
        
        # Verificar memória disponível
        mem_info = self._check_gpu_memory()
        print(f"\n💾 Memória GPU: {mem_info['free']:.1f} GB livres de {mem_info['total']:.1f} GB")
        
        if mem_info['free'] < 3.0:
            raise RuntimeError(
                f"Memória GPU insuficiente ({mem_info['free']:.1f} GB livres). "
                "Feche outros programas usando GPU."
            )
        
        # Inicializar pipeline se necessário
        if self.svd_pipeline is None:
            self._init_svd_pipeline(progress_callback=progress_callback)
        
        # Carregar imagem
        if isinstance(image, str):
            input_image = load_image(image)
        else:
            input_image = image
        
        # Redimensionar para resolução otimizada (economiza MUITA memória)
        original_size = input_image.size
        if input_image.size != resolution:
            print(f"📐 Redimensionando imagem: {original_size} → {resolution}")
            input_image = input_image.resize(resolution, Image.Resampling.LANCZOS)
        
        if progress_callback:
            progress_callback(0.4, f"🎬 Gerando {num_frames} frames com SVD...")
        else:
            print(f"\n🎬 Gerando vídeo com SVD...")
            print(f"  Frames: {num_frames}")
            print(f"  Resolução: {resolution}")
            print(f"  Steps: {num_inference_steps}")
            print(f"  FPS: {fps}")
        
        try:
            # Limpar cache antes de gerar
            torch.cuda.empty_cache()
            
            if progress_callback:
                progress_callback(0.5, f"🎨 Processando {num_inference_steps} passos de inferência...")
            
            # Gerar frames
            frames = self.svd_pipeline(
                input_image,
                decode_chunk_size=decode_chunk_size,  # Processar 1 frame por vez
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                motion_bucket_id=motion_bucket_id,
                fps=fps
            ).frames[0]
            
            if progress_callback:
                progress_callback(0.85, "📹 Processando frames do vídeo...")
            
            # Limpar cache após gerar
            torch.cuda.empty_cache()
            
            # Exportar para vídeo
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if progress_callback:
                progress_callback(0.9, "💾 Salvando vídeo...")
            
            export_to_video(frames, str(output_file), fps=fps)
            
            if progress_callback:
                progress_callback(0.95, "✅ Vídeo salvo com sucesso!")
            else:
                print(f"\n✅ Vídeo gerado com sucesso!")
                print(f"  Salvo em: {output_file}")
                print(f"  Duração: ~{num_frames/fps:.1f} segundos")
            
            # Salvar metadados
            if save_metadata:
                self.video_params = {
                    "method": "stable_video_diffusion",
                    "num_frames": num_frames,
                    "fps": fps,
                    "resolution": f"{resolution[0]}x{resolution[1]}",
                    "original_resolution": f"{original_size[0]}x{original_size[1]}",
                    "num_inference_steps": num_inference_steps,
                    "motion_bucket_id": motion_bucket_id,
                    "decode_chunk_size": decode_chunk_size,
                    "duration": num_frames / fps,
                    "gpu_memory_used": f"{mem_info['allocated']:.1f} GB",
                    "timestamp": datetime.now().isoformat()
                }
                
                metadata_path = output_file.parent / "svd_metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(self.video_params, f, indent=2, ensure_ascii=False)
                print(f"  Metadados: {metadata_path}")
            
            return str(output_file)
            
        except torch.cuda.OutOfMemoryError as e:
            # Limpar tudo e sugerir fallback
            torch.cuda.empty_cache()
            if self.svd_pipeline is not None:
                del self.svd_pipeline
                self.svd_pipeline = None
            
            raise RuntimeError(
                f"❌ Memória GPU insuficiente (OOM). "
                f"Tente reduzir:\n"
                f"  - Resolução: {resolution} → (384, 256)\n"
                f"  - Frames: {num_frames} → 10\n"
                f"  - Steps: {num_inference_steps} → 20"
            ) from e
    
    def cleanup_svd(self):
        """
        Limpa pipeline SVD da memória GPU
        """
        if self.svd_pipeline is not None:
            del self.svd_pipeline
            self.svd_pipeline = None
            torch.cuda.empty_cache()
            print("🧹 Pipeline SVD removido da memória GPU")


# Exemplo de uso
if __name__ == "__main__":
    from glob import glob

    # Criar gerador
    generator = VideoGenerator()

    # Buscar imagens geradas
    image_files = sorted(glob("outputs/images/character_*.png"))

    if len(image_files) > 0:
        print(f"Encontradas {len(image_files)} imagens")

        # Criar vídeo
        video_path = generator.create_video_from_images(
            images=image_files,
            output_path="outputs/videos/character_animation.mp4",
            fps=3,
            duration_per_image=1.5,
            transition_frames=15,
            add_loop=True
        )

        print(f"\nVídeo criado: {video_path}")
    else:
        print("Nenhuma imagem encontrada em outputs/images/")
        print("Execute primeiro o image_generator.py")
