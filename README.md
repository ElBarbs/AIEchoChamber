# Recursive Vision

A project that explores the concept of an AI echo chamber by iteratively transforming images through AI-generated descriptions and back to images again.

## Overview

This project implements an iterative process:
1. Start with an initial image
2. Generate a detailed description using vision models
3. Create a new image based on that description using text-to-image models
4. Repeat steps 2-3 multiple times to observe how the image evolves

## Requirements

- Python 3.7+
- PyTorch with CUDA (for faster generation)
- Access to OpenAI API for advanced vision models (optional)
- Various Python libraries (see requirements.txt)

## Project Structure

- `main.py`: The main entry point script that orchestrates the echo chamber process
- `echo_chamber.py`: Core implementation of the echo chamber algorithm
- `config.py`: Configuration settings for models and parameters
- `image_utils.py`: Functions for handling images and processing
- `models.py`: Model loading and inference implementations
- `.env`: Environment variables (API keys, etc.) - not tracked in git
- `requirements.txt`: List of required Python packages

## Supported Models

### Text-to-Image Models
- Stable Diffusion XL
- Stable Diffusion 3
- Stable Diffusion 3.5
- FLUX.1-schnell
- FLUX.1-dev

### Vision Models
- Florence-2-large
- GPT-4o (via OpenAI API)

## Setup & Running

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AIEchoChamber

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables by creating a .env file:
OPENAI_API_KEY=your_api_key_here
```

### Command-line Usage

```bash
# Basic usage
python main.py --input /path/to/input/image.jpg --output /path/to/output/directory --iterations 10

# Run with specific models
python main.py --input friends.jpg --output results --img_model stable-diffusion-3 --vision_model florence2

# Or use run_job.sh for predefined settings
bash run_job.sh
```

## Output

For each iteration, the script generates and saves:

- The generated image (`iteration_X_image.png`)
- The image description (`iteration_X_description.txt`)
- A log of the full process

## Server Usage

This script is designed to run on a server without requiring a GUI:

- It accepts command-line arguments for specifying input, output, and iterations
- It doesn't generate any visual previews during execution
- Output is saved to disk for later examination
- Progress is reported via console logging

## Notes

- The first run will download the required models (several GB)
- You will need to provide your OpenAI API key in a `.env` file or environment variable if using OpenAI models
- Generation can be time-consuming, especially with many iterations
- Ensure you have sufficient disk space for the models and generated images
