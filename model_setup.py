"""
Model Setup Module
Contains functions for loading models and setting up API clients
"""

import os
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
from openai import OpenAI


def load_models(config):
    """
    Loads Stable Diffusion model and Compel processor

    Args:
        config: Dictionary containing model configuration parameters

    Returns:
        tuple: (DiffusionPipeline, Compel) - The loaded models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Stable Diffusion model (this may take a few minutes)...")

    # Load Stable Diffusion with specific precision settings
    text2img_pipe = DiffusionPipeline.from_pretrained(
        config["model_checkpoint"],
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        use_safetensors=True,
        scheduler=EulerDiscreteScheduler(**config["scheduler_kwargs"]),
    ).to(device)

    # Initialize compel
    compel_proc = Compel(
        tokenizer=[text2img_pipe.tokenizer, text2img_pipe.tokenizer_2],
        text_encoder=[text2img_pipe.text_encoder,
                      text2img_pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True]
    )

    print("Model loaded successfully!")
    return text2img_pipe, compel_proc


def setup_openai_client():
    """
    Sets up OpenAI client using API key from environment or manual input

    Returns:
        OpenAI: Configured OpenAI client
    """
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        # Fallback to manual input
        print("Please enter your OpenAI API key:")
        api_key = input()
        # Save to environment variable for future use in the session
        os.environ['OPENAI_API_KEY'] = api_key

    client = OpenAI(api_key=api_key)
    return client
