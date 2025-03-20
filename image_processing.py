"""
Image Processing Module
Contains functions for processing and describing images
"""

import io
import base64
from PIL import Image


def load_initial_image(image_path):
    """
    Loads and preprocesses the initial image

    Args:
        image_path: Path to the input image file

    Returns:
        PIL.Image: The processed initial image
    """
    # Open the provided image
    image = Image.open(image_path)

    # Resize image to work with the model
    max_size = 512
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


def generate_structured_description(image, openai_client):
    """
    Generate a detailed structured description of the image using GPT-4o.

    Args:
        image: PIL Image object
        openai_client: OpenAI client

    Returns:
        str: Detailed description of the image.
    """
    # Convert image to base64 for API
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Call OpenAI API with GPT-4o
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
              "role": "developer",
              "content": """You are an image descriptor specialized in providing detailed, accurate descriptions that allow others to visualize and potentially reproduce images. When presented with an image, analyze and describe:

                            Overall composition and subject matter
                            Key visual elements in order of prominence
                            Spatial relationships and positioning of objects
                            Color palette, lighting, and atmosphere
                            Style, medium, and artistic techniques (if applicable)
                            Textures, patterns, and materials
                            Perspective and depth
                            Scale and proportions of elements
                            Important details that contribute to the image's uniqueness

                            Provide descriptions that are methodical and comprehensive, moving from general observations to specific details. Use precise, descriptive language that avoids ambiguity. Your goal is to create a verbal representation that would enable someone who cannot see the image to mentally reconstruct it or reproduce it artistically."""
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=1.25,
        max_completion_tokens=500
    )

    return response.choices[0].message.content
