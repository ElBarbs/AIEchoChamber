"""
AI Echo Chamber - Main Script
This script orchestrates the AI Echo Chamber process by running multiple iterations
of image-to-description-to-image transformations.
"""

import random
import torch
import argparse
import json
from pathlib import Path

from model_setup import load_models, setup_openai_client
from image_processing import load_initial_image, generate_structured_description

# Set up configuration.
config = {
    "model_weights": "../huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
    "guidance_scale_min": 5,
    "guidance_scale_max": 8,
    "num_inference_steps": 25,
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

# Main recursive echo chamber process.


def run_echo_chamber(initial_image, openai_client, output_dir, iterations=3):
    """
    Run the AI Echo Chamber process for a specified number of iterations.

    Args:
        initial_image: PIL Image object to start with
        openai_client: OpenAI client for image description
        output_dir: Directory to save generated images and descriptions
        iterations: Number of iterations to run

    Returns:
        dict: Dictionary containing results for each iteration
    """
    text2img_pipe, compel_proc = load_models(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory if it doesn't exist.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a JSONL file to store all descriptions.
    descriptions_file = output_dir / "descriptions.jsonl"

    # Save the initial image.
    initial_image_path = output_dir / "iteration_0_input.png"
    initial_image.save(initial_image_path)
    print(f"Saved initial image to {initial_image_path}")

    results = {
        0: {"image": initial_image, "description": None}
    }

    # Generate description for initial image.
    current_description = generate_structured_description(
        initial_image, openai_client)
    results[0]["description"] = current_description

    # Save initial description to JSONL file.
    with open(descriptions_file, "w") as f:
        json.dump({
            "iteration": 0,
            "type": "input",
            "description": current_description,
            "image_path": str(initial_image_path)
        }, f)
        f.write("\n")

    print(f"Saved initial description to {descriptions_file}")

    for i in range(1, iterations + 1):
        print(f"\nProcessing iteration {i}/{iterations}...")

        # Use the full description as the prompt.
        results[i] = {"image": None, "description": None}

        # Process the prompt with compel.
        prompt_embeds, pooled_embeds = compel_proc(current_description)
        neg_prompt_embeds, neg_pooled_embeds = compel_proc(
            "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature")
        prompt_embeds, neg_prompt_embeds = compel_proc.pad_conditioning_tensors_to_same_length(
            [prompt_embeds, neg_prompt_embeds])

        # Generate a new image using the processed prompt embeddings.
        output = text2img_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_pooled_prompt_embeds=neg_pooled_embeds,
            output_type="pil",
            guidance_scale=random.uniform(
                config["guidance_scale_min"], config["guidance_scale_max"]),
            num_inference_steps=config["num_inference_steps"],
        )

        new_image = output.images[0]
        results[i]["image"] = new_image

        # Save the generated image.
        image_path = output_dir / f"iteration_{i}_image.png"
        new_image.save(image_path)
        print(f"Saved generated image to {image_path}")

        # Generate a new description based on the new image.
        current_description = generate_structured_description(
            new_image, openai_client)
        results[i]["description"] = current_description

        # Append the description to the JSONL file.
        with open(descriptions_file, "a") as f:
            json.dump({
                "iteration": i,
                "type": "generated",
                "description": current_description,
                "image_path": str(image_path),
                # Include the prompt that generated this image
                "prompt": results[i-1]["description"]
            }, f)
            f.write("\n")

        print(f"Appended description to {descriptions_file}")
        print(f"Completed iteration {i}/{iterations}")

    return results


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='AI Echo Chamber: Iterative image-to-description-to-image transformation')
    parser.add_argument('--input', '-i', type=str,
                        required=True, help='Path to the input image')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Directory to save generated images and descriptions')
    parser.add_argument('--iterations', '-n', type=int,
                        default=20, help='Number of iterations to run')
    args = parser.parse_args()

    # Print device information.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up OpenAI client.
    openai_client = setup_openai_client()

    # Load initial image.
    initial_image = load_initial_image(args.input)
    print(f"Loaded initial image from {args.input}")

    # Run the echo chamber process.
    print(f"Will run {args.iterations} iterations of the AI Echo Chamber")
    print(f"Saving results to {args.output}")

    run_echo_chamber(
        initial_image,
        openai_client,
        output_dir=args.output,
        iterations=args.iterations
    )

    print(f"All iterations completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
