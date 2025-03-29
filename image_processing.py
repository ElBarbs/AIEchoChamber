"""
Image Processing Module
Contains functions for processing and describing images.
"""

import io
import base64
from PIL import Image


def load_initial_image(image_path):
    """
    Loads and preprocesses the initial image.

    Args:
        image_path: Path to the input image file.

    Returns:
        PIL.Image: The processed initial image.
    """
    # Open the provided image.
    image = Image.open(image_path)

    # Resize image to work with the model.
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
    Generate a detailed structured description of the image using GPT-4o-mini.

    Args:
        image: PIL Image object.
        openai_client: OpenAI client.

    Returns:
        str: Detailed description of the image.
    """
    # Convert image to base64 for API.
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Call OpenAI API with GPT-4o-mini.
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
              "role": "developer",
              "content": """You are an image descriptor with a focus on creative and varied descriptions. When presented with an image:

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
                            
                            Important: Do not include any meta-commentary about your description process. Do not add closing statements about the description's purpose or effectiveness. Simply describe the image and end with the title."""
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
        max_completion_tokens=600
    )

    return response.choices[0].message.content
