"""
AI Echo Chamber - Main Script
This script orchestrates the AI Echo Chamber process by running multiple iterations
of image-to-description-to-image transformations.
"""

import torch
import argparse
from pathlib import Path

from model_setup import load_models, setup_openai_client
from image_processing import load_initial_image, generate_structured_description

# Set up configuration.
config = {
    "model_weights": "/home/lbarbier/projects/def-vigliens/lbarbier/AIEchoChamber/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0",
    "seed": 1999,
    "guidance_scale": 1.5,
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
    generator = [torch.Generator(device=str(
        device)).manual_seed(config["seed"])]

    # Create output directory if it doesn't exist.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Save initial description.
    with open(output_dir / "iteration_0_description.txt", "w") as f:
        f.write(current_description)
    print(
        f"Saved initial description to {output_dir / 'iteration_0_description.txt'}")

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
            guidance_scale=config["guidance_scale"],
            num_inference_steps=config["num_inference_steps"],
            generator=generator,
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

        # Save the description.
        description_path = output_dir / f"iteration_{i}_description.txt"
        with open(description_path, "w") as f:
            f.write(current_description)
        print(f"Saved description to {description_path}")

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
