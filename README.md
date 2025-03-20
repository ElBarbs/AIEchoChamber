# AI Echo Chamber

A project that explores the concept of an AI echo chamber by iteratively transforming images through AI-generated descriptions and back to images again.

## Overview

This project implements an iterative process:
1. Start with an initial image
2. Generate a detailed description using GPT-4o
3. Create a new image based on that description using Stable Diffusion
4. Repeat steps 2-3 multiple times to observe how the image evolves

## Requirements

- Python 3.7+
- PyTorch with CUDA (for faster generation)
- Access to OpenAI API (GPT-4o)
- Various Python libraries (see requirements.txt)

## Project Structure

- `main.py`: The main script that orchestrates the echo chamber process
- `model_setup.py`: Functions for loading models and setting up API clients
- `image_processing.py`: Functions for handling images and generating descriptions
- `requirements.txt`: List of required Python packages

## Setup & Running

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-echo-chamber

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here  # On Windows: set OPENAI_API_KEY=your_api_key_here
```

### Command-line Usage

```bash
# Basic usage
python main.py --input /path/to/input/image.jpg --output /path/to/output/directory --iterations 10

# Arguments:
# --input, -i : Path to the input image (required)
# --output, -o : Directory to save results (default: 'output')
# --iterations, -n : Number of iterations to run (default: 20)
```

## Output

For each iteration, the script generates and saves:
- The generated image (`iteration_X_image.png`)
- The image description (`iteration_X_description.txt`)

## Server Usage

This script is designed to run on a server without requiring a GUI:
- It accepts command-line arguments for specifying input, output, and iterations
- It doesn't generate any visual previews during execution
- Output is saved to disk for later examination
- Progress is reported via console logging

## Notes

- The first run will download the Stable Diffusion model (~7GB)
- You will need to provide your OpenAI API key if it's not set in your environment
- Generation can be time-consuming, especially with many iterations
- Ensure you have sufficient disk space for the model and generated images