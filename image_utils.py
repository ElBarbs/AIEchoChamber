"""
Image Processing Module
Contains functions for processing and describing images.
"""

import torch
import io
import base64
from PIL import Image

from config import IMAGE_CONFIG, MODEL_CONFIG, IMAGE_DESCRIPTION_PROMPT


def load_initial_image(image_path):
    """
    Loads and preprocesses the initial image.

    Args:
        image_path: Path to the input image file.

    Returns:
        PIL.Image: The processed initial image.
    """
    try:
        # Open the provided image
        image = Image.open(image_path)

        # Resize image to work with the model
        max_size = IMAGE_CONFIG["max_size"]
        image = image.convert("RGB")
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        image = image.resize((new_width, new_height))

        return image

    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        raise


def encode_image_to_base64(image):
    """
    Converts a PIL image to base64 encoding for API calls.

    Args:
        image: PIL Image object

    Returns:
        str: Base64-encoded image string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=IMAGE_CONFIG["format"])
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_description(image, vision_model=None, vision_processor=None, openai_client=None, description_type="florence2"):
    """
    Generate a description of the image using either Florence2 or GPT-4o-mini.

    Args:
        image: PIL Image object
        vision_model: Optional Florence2 model
        vision_processor: Optional Florence2 processor
        openai_client: Optional OpenAI client
        description_type: 'florence2' or 'gpt4o-mini'

    Returns:
        str: Description of the image
    """
    if description_type == "gpt4o-mini":
        if openai_client is None:
            raise ValueError(
                "OpenAI client required for GPT-4o-mini descriptions")
        return _generate_gpt4_description(image, openai_client)
    else:
        if vision_model is None or vision_processor is None:
            raise ValueError(
                "Vision model and processor required for Florence2 descriptions")
        return _generate_florence2_description(image, vision_model, vision_processor)


def _generate_florence2_description(image, vision_model, vision_processor):
    """
    Generate a description using the Florence2 vision model.

    Args:
        image: PIL Image object
        vision_model: Florence2 model
        vision_processor: Florence2 processor

    Returns:
        str: Description of the image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    task = MODEL_CONFIG["vision_models"]["florence2"]["task"]

    inputs = vision_processor(
        text=task,
        images=image,
        return_tensors="pt"
    ).to(device, torch_dtype)

    generated_ids = vision_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=77,  # 4096 tokens for 1.5B model
        num_beams=3,
        do_sample=False
    )

    generated_text = vision_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed_answer = vision_processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )

    return parsed_answer[task]


def _generate_gpt4_description(image, openai_client):
    """
    Generate a description using GPT-4o-mini.

    Args:
        image: PIL Image object
        openai_client: OpenAI client

    Returns:
        str: Description of the image
    """
    # Convert image to base64 for API
    base64_image = encode_image_to_base64(image)

    # Get configuration values
    model_config = MODEL_CONFIG["vision_models"]["gpt4o-mini"]

    # Call OpenAI API with GPT-4o-mini
    response = openai_client.chat.completions.create(
        model=model_config["model_id"],
        messages=[
            {
                "role": "developer",
                "content": IMAGE_DESCRIPTION_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=model_config["temperature"],
        max_completion_tokens=model_config["max_tokens"]
    )

    return response.choices[0].message.content
