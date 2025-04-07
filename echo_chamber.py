"""
Echo Chamber Module
Core logic for the AI Echo Chamber process.
"""

import random
import json
from pathlib import Path

from config import MODEL_CONFIG, DEFAULT_NEGATIVE_PROMPT
from models import model_manager
from image_utils import generate_description


class EchoChamber:
    """
    Class handling the main Echo Chamber process - running multiple iterations
    of image-to-description-to-image transformations.
    """

    def __init__(self, initial_image, output_dir, description_type="florence2", txt2img_model=None):
        """
        Initialize the Echo Chamber process.

        Args:
            initial_image: PIL Image to start with
            output_dir: Directory to save outputs
            description_type: Type of description model to use ('florence2' or 'gpt4o-mini')
        """
        self.initial_image = initial_image
        self.output_dir = Path(output_dir)
        self.description_type = description_type
        self.results = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set current text-to-image model in config
        if txt2img_model:
            MODEL_CONFIG["current_txt2img_model"] = txt2img_model
        else:
            MODEL_CONFIG["current_txt2img_model"] = MODEL_CONFIG["default_txt2img_model"]

        # Initialize description file
        self.descriptions_file = self.output_dir / "descriptions.jsonl"

        # Load models based on description type
        if description_type == "florence2":
            # We'll use the Florence2 model
            self.vision_model, self.vision_processor = model_manager.vision_model
            self.openai_client = None
        else:
            # We'll use the OpenAI GPT-4o-mini
            self.openai_client = model_manager.openai_client
            self.vision_model = None
            self.vision_processor = None

        # Load text-to-image model
        self.text2img_pipe, self.compel_proc = model_manager.text2img_model

        # Get model info
        model_name = MODEL_CONFIG["current_txt2img_model"]
        if self.compel_proc is None:
            print(f"Using {model_name} without Compel processing.")
        else:
            print(f"Using {model_name} with Compel processing.")

    def run(self, iterations=1):
        """
        Run the Echo Chamber process for the specified number of iterations.

        Args:
            iterations: Number of iterations to perform

        Returns:
            dict: Results of the process
        """
        print(f"Starting Echo Chamber process with {iterations} iterations...")

        # Save the initial image
        initial_image_path = self.output_dir / "iteration_0_input.png"
        self.initial_image.save(initial_image_path)
        print(f"Saved initial image to {initial_image_path}")

        # Initialize results with the initial image
        self.results = {
            0: {"image": self.initial_image, "description": None}
        }

        # Generate description for initial image
        current_description = generate_description(
            self.initial_image,
            self.vision_model,
            self.vision_processor,
            self.openai_client,
            self.description_type
        )

        self.results[0]["description"] = current_description

        # Save initial description
        self._save_description(0, current_description,
                               str(initial_image_path), "input")

        # Run iterations
        for i in range(1, iterations + 1):
            print(f"\nProcessing iteration {i}/{iterations}...")
            self._run_iteration(i, current_description)
            current_description = self.results[i]["description"]

        return self.results

    def _run_iteration(self, iteration, prompt):
        """
        Run a single iteration of the echo chamber process.

        Args:
            iteration: Current iteration number
            prompt: Text prompt to generate image from
        """
        # Initialize result entry
        self.results[iteration] = {"image": None, "description": None}

        # Generate image from description
        new_image = self._generate_image_from_prompt(prompt)
        self.results[iteration]["image"] = new_image

        # Save generated image
        image_path = self.output_dir / f"iteration_{iteration}_image.png"
        new_image.save(image_path)
        print(f"Saved generated image to {image_path}")

        # Generate description from new image
        new_description = generate_description(
            new_image,
            self.vision_model,
            self.vision_processor,
            self.openai_client,
            self.description_type
        )

        self.results[iteration]["description"] = new_description

        # Save description
        self._save_description(
            iteration,
            new_description,
            str(image_path),
            "generated",
            prompt
        )

        print(f"Completed iteration {iteration}")

    def _generate_image_from_prompt(self, prompt):
        """
        Generate an image from a text prompt using Stable Diffusion.

        Args:
            prompt: Text description to generate from

        Returns:
            PIL.Image: Generated image
        """
        # Generate image
        if self.compel_proc is None:
            # Model without Compel - pass prompt directly
            output = self.text2img_pipe(
                prompt=prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                output_type="pil",
                guidance_scale=random.uniform(
                    MODEL_CONFIG["guidance_scale_min"],
                    MODEL_CONFIG["guidance_scale_max"]
                ),
                num_inference_steps=MODEL_CONFIG["num_inference_steps"],
            )
        else:
            # Process the prompt with compel
            prompt_embeds, pooled_embeds = self.compel_proc(prompt)
            neg_prompt_embeds, neg_pooled_embeds = self.compel_proc(
                DEFAULT_NEGATIVE_PROMPT)

            # Pad embeddings to same length
            prompt_embeds, neg_prompt_embeds = self.compel_proc.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, neg_prompt_embeds]
            )

            output = self.text2img_pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_embeds,
                output_type="pil",
                guidance_scale=random.uniform(
                    MODEL_CONFIG["guidance_scale_min"],
                    MODEL_CONFIG["guidance_scale_max"]
                ),
                num_inference_steps=MODEL_CONFIG["num_inference_steps"],
            )

        return output.images[0]

    def _save_description(self, iteration, description, image_path, type_label, prompt=None):
        """
        Save a description to the JSONL file.

        Args:
            iteration: Current iteration number
            description: Text description
            image_path: Path to the corresponding image
            type_label: Type of description ('input' or 'generated')
            prompt: Optional prompt that generated this image
        """
        data = {
            "iteration": iteration,
            "type": type_label,
            "description": description,
            "image_path": image_path
        }

        if prompt:
            data["prompt"] = prompt

        mode = "w" if iteration == 0 else "a"

        with open(self.descriptions_file, mode) as f:
            json.dump(data, f)
            f.write("\n")

        print(
            f"{'Saved' if iteration == 0 else 'Appended'} description to {self.descriptions_file}")
