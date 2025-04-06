"""
AI Echo Chamber - Main Script
Entry point for the AI Echo Chamber application that iteratively 
transforms images through AI description and generation cycles.
"""

import argparse
import torch
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, DEFAULT_ITERATIONS
from image_utils import load_initial_image
from echo_chamber import EchoChamber


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='AI Echo Chamber: Iterative image-to-description-to-image transformation'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input image'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Directory to save generated images and descriptions (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f'Number of iterations to run (default: {DEFAULT_ITERATIONS})'
    )

    parser.add_argument(
        '--vision', '-v',
        type=str,
        default='florence2',
        choices=['florence2', 'gpt4o-mini'],
        help='Vision model to use for descriptions (default: florence2)'
    )

    return parser.parse_args()


def print_system_info():
    """Print information about the system and available devices."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"System information:")
    print(f"- Using device: {device}")

    if device.type == 'cuda':
        print(f"- CUDA version: {torch.version.cuda}")
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"- Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    print_system_info()

    try:
        # Convert output path to Path object
        output_dir = Path(args.output)

        # Create a unique directory for this run based on the input filename
        input_path = Path(args.input)
        run_name = input_path.stem
        run_output_dir = output_dir / run_name

        # Load initial image
        print(f"Loading initial image from {args.input}")
        initial_image = load_initial_image(args.input)

        # Create and run the echo chamber
        echo_chamber = EchoChamber(
            initial_image=initial_image,
            output_dir=run_output_dir,
            description_type=args.vision
        )

        print(f"Starting Echo Chamber with {args.iterations} iterations")
        print(f"Using {args.vision} for image description")
        print(f"Saving results to {run_output_dir}")

        results = echo_chamber.run(iterations=args.iterations)

        print(f"All iterations completed. Results saved to {run_output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
