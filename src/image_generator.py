"""
Módulo de Geração de Imagens
Pipeline para criar imagens consistentes de personagens usando Stable Diffusion

Instituição: Universidade de Pernambuco (UPE)
Programa: Residência em IA Generativa
Disciplina: IA Generativa para Mídia Visual
Autores: Vanthuir Maia e Rodrigo Santana
"""

import os
import random
from typing import List, Optional, Dict
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import json
from datetime import datetime


class ImageGenerator:
    """
    Gerador de imagens usando Stable Diffusion com controle de consistência
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        use_fp16: bool = True
    ):
        """
        Inicializa o gerador de imagens

        Args:
            model_id: ID do modelo no Hugging Face
            device: Dispositivo para executar (cuda, cpu, ou auto)
            use_fp16: Usar precisão FP16 para economizar memória
        """
        self.model_id = model_id

        # Detectar dispositivo automaticamente
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.pipe = None
        self.generation_params = {}

    def load_model(self):
        """Carrega o modelo Stable Diffusion"""
        print(f"Carregando modelo {self.model_id} no dispositivo {self.device}...")

        # Configurar tipo de dados
        torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        # Carregar pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Desabilitar safety checker para maior controle
            requires_safety_checker=False
        )

        # Configurar scheduler para melhor qualidade
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Mover para dispositivo
        self.pipe = self.pipe.to(self.device)

        # Habilitar otimizações
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            # Descomentar se tiver GPU com pouca memória
            # self.pipe.enable_vae_slicing()

        print("Modelo carregado com sucesso!")

    def generate_images(
        self,
        prompt: str,
        num_images: int = 10,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        width: int = 512,
        height: int = 512,
        output_dir: str = "outputs/images",
        save_metadata: bool = True,
        progress_callback = None
    ) -> List[Image.Image]:
        """
        Gera múltiplas imagens do personagem com consistência visual

        Args:
            prompt: Descrição do personagem
            num_images: Número de imagens a gerar
            negative_prompt: Prompt negativo (o que evitar)
            seed: Seed para reprodutibilidade (se None, usa aleatório)
            guidance_scale: Força de aderência ao prompt (7-15 recomendado)
            num_inference_steps: Passos de difusão (mais = melhor qualidade)
            width: Largura da imagem
            height: Altura da imagem
            output_dir: Diretório para salvar imagens
            save_metadata: Salvar metadados da geração

        Returns:
            Lista de imagens PIL geradas
        """
        if self.pipe is None:
            self.load_model()

        # Criar diretório de saída
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Definir seed base para consistência
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Prompt negativo padrão para melhor qualidade
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, ugly, "
                "bad anatomy, bad proportions, extra limbs, "
                "text, watermark, signature"
            )

        # Armazenar parâmetros de geração
        self.generation_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat()
        }

        images = []
        seeds_used = []

        print(f"Gerando {num_images} imagens...")
        print(f"Prompt: {prompt}")
        print(f"Seed base: {seed}")

        import time
        start_time = time.time()

        for i in range(num_images):
            # Usar seeds sequenciais para manter consistência
            current_seed = seed + i
            seeds_used.append(current_seed)

            # Configurar gerador
            generator = torch.Generator(device=self.device).manual_seed(current_seed)

            # Gerar imagem
            print(f"Gerando imagem {i+1}/{num_images} (seed: {current_seed})...")

            # Callback de progresso antes de gerar
            if progress_callback:
                progress_callback(i, num_images, "generating")

            img_start = time.time()

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )

            image = result.images[0]
            images.append(image)

            # Salvar imagem
            image_filename = f"character_{i+1:03d}_seed{current_seed}.png"
            image_path = output_path / image_filename
            image.save(image_path)
            print(f"  Salva: {image_filename}")

            # Calcular tempo e estimar tempo restante
            img_time = time.time() - img_start
            elapsed = time.time() - start_time
            avg_time_per_image = elapsed / (i + 1)
            remaining_images = num_images - (i + 1)
            estimated_remaining = avg_time_per_image * remaining_images

            # Callback de progresso após gerar
            if progress_callback:
                progress_callback(
                    i + 1,
                    num_images,
                    "completed",
                    time_remaining=estimated_remaining,
                    time_per_image=img_time
                )

        # Salvar metadados
        if save_metadata:
            self.generation_params["seeds_used"] = seeds_used
            self.generation_params["num_images"] = num_images

            metadata_path = output_path / "generation_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.generation_params, f, indent=2, ensure_ascii=False)
            print(f"\nMetadados salvos em: {metadata_path}")

        print(f"\n{num_images} imagens geradas com sucesso!")
        return images

    def get_generation_params(self) -> Dict:
        """Retorna os parâmetros da última geração"""
        return self.generation_params.copy()

    def cleanup(self):
        """Libera memória do modelo"""
        if self.pipe is not None:
            del self.pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.pipe = None
            print("Modelo removido da memória")


# Exemplo de uso
if __name__ == "__main__":
    # Criar gerador
    generator = ImageGenerator()

    # Prompt de exemplo
    prompt = (
        "A cute cartoon robot character, round body, big expressive eyes, "
        "friendly smile, blue and white colors, simple design, "
        "mascot style, standing pose, white background, "
        "digital art, high quality, consistent character design"
    )

    # Gerar imagens
    images = generator.generate_images(
        prompt=prompt,
        num_images=10,
        seed=42,
        guidance_scale=7.5,
        num_inference_steps=50
    )

    print(f"\nGeradas {len(images)} imagens!")
