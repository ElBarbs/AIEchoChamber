"""
Configuration Module
Central configuration for the AI Echo Chamber project.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# Model configuration
MODEL_CONFIG = {
    "txt2img_weights": "../huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    "vision_models": {
        "florence2": {
            "model_id": "microsoft/Florence-2-large",
            "task": "<MORE_DETAILED_CAPTION>"
        },
        "gpt4o-mini": {
            "model_id": "gpt-4o-mini",
            "max_tokens": 600,
            "temperature": 1.25
        }
    },
    "guidance_scale_min": 5,
    "guidance_scale_max": 8,
    "num_inference_steps": 50,
    "scheduler_kwargs": {
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "interpolation_type": "linear",
        "num_train_timesteps": 1000,
        "prediction_type": "epsilon",
        "steps_offset": 1,
        "timestep_spacing": "leading",
        "trained_betas": None,
        "use_karras_sigmas": False,
    }
}

# Default negative prompt for stable diffusion
DEFAULT_NEGATIVE_PROMPT = """
lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, 
ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, 
poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, 
bad anatomy, bad proportions, extra limbs, cloned face, disfigured, 
gross proportions, malformed limbs, missing arms, missing legs, extra arms, 
extra legs, fused fingers, too many fingers, long neck, username, watermark, signature
"""

# Image processing
IMAGE_CONFIG = {
    "max_size": 512,  # Maximum dimension for resized images
    "format": "JPEG"  # Format for base64 encoding
}

# GPT4o-mini prompt template
IMAGE_DESCRIPTION_PROMPT = """
You are an image descriptor with a focus on creative and varied descriptions. When presented with an image:

1. Identify the core subject matter, but consider multiple interpretations.
2. Highlight 3-5 key visual elements, prioritizing unusual or distinctive features.
3. Describe the overall mood and emotional impact before technical details.
4. Vary your descriptive vocabulary.
5. Consider multiple artistic styles or genres this image might represent.
6. Describe colors in terms of associations rather than just naming them.
7. Include one unexpected or surprising observation about the image.
8. Limit your description to 3-4 paragraphs maximum.
9. Use natural language rather than formal analytical structure.
10. End with a brief creative title that captures the essence of the image.

Your goal is to provide a fresh, insightful perspective on each image that inspires imagination rather than technical reproduction.

Important: Do not include any meta-commentary about your description process. Do not add closing statements about the description's purpose or effectiveness. Simply describe the image and end with the title.
"""

# Path to .env file for environment variables
ENV_FILE = BASE_DIR / ".env"

# Default number of iterations
DEFAULT_ITERATIONS = 20
