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


class VideoGenerator:
    """
    Gerador de vídeos animados a partir de imagens com transições suaves
    """

    def __init__(self):
        """Inicializa o gerador de vídeo"""
        self.video_params = {}

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
