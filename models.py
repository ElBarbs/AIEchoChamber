"""
Models Module
Contains functions for loading models and setting up API clients.
"""

import os
import torch
from dotenv import load_dotenv
from pathlib import Path
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from transformers import AutoModelForCausalLM, AutoProcessor
from compel import Compel, ReturnedEmbeddingsType
from openai import OpenAI

from config import MODEL_CONFIG, ENV_FILE


class ModelManager:
    """Manager class for handling model loading and caching."""

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        self._text2img_pipe = None
        self._compel_proc = None
        self._vision_model = None
        self._vision_processor = None
        self._openai_client = None

    @property
    def text2img_model(self):
        """Lazy-load text2img model on first access."""
        if self._text2img_pipe is None:
            self._load_text2img_model()
        return self._text2img_pipe, self._compel_proc

    @property
    def vision_model(self):
        """Lazy-load vision model on first access."""
        if self._vision_model is None:
            self._load_vision_model()
        return self._vision_model, self._vision_processor

    @property
    def openai_client(self):
        """Lazy-load OpenAI client on first access."""
        if self._openai_client is None:
            self._setup_openai_client()
        return self._openai_client

    def _load_text2img_model(self):
        """Loads text-to-image model based on configuration."""
        # Get current model configuration
        model_name = MODEL_CONFIG.get(
            "current_txt2img_model", MODEL_CONFIG["default_txt2img_model"])
        model_config = MODEL_CONFIG["txt2img_models"][model_name]
        model_id = model_config["model_id"]

        print(
            f"Loading {model_name} model ({model_id}) on {self.device} (this may take a few minutes)...")

        # Set up pipeline args
        pipe_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
        }

        # Add custom scheduler if specified
        if model_config.get("custom_scheduler", False):
            scheduler_type = model_config.get(
                "scheduler", "EulerDiscreteScheduler")
            scheduler_kwargs = model_config.get("scheduler_kwargs", {})

            # Create the appropriate scheduler
            if scheduler_type == "EulerDiscreteScheduler":
                scheduler = EulerDiscreteScheduler(**scheduler_kwargs)
            else:
                # Default to EulerDiscreteScheduler if unknown type
                scheduler = EulerDiscreteScheduler(**scheduler_kwargs)

            pipe_kwargs["scheduler"] = scheduler

        # Load the model
        self._text2img_pipe = DiffusionPipeline.from_pretrained(
            model_id,
            **pipe_kwargs
        ).to(self.device)

        # Initialize Compel if needed
        self._compel_proc = None
        if model_config.get("use_compel", False):
            self._compel_proc = Compel(
                tokenizer=[self._text2img_pipe.tokenizer,
                           self._text2img_pipe.tokenizer_2],
                text_encoder=[self._text2img_pipe.text_encoder,
                              self._text2img_pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )

        print(f"{model_name} model loaded successfully!")

    def _load_vision_model(self):
        """Loads the vision model for image processing."""
        model_id = MODEL_CONFIG["vision_models"]["florence2"]["model_id"]
        print(f"Loading vision model {model_id} on {self.device}...")

        try:
            self._vision_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)

            self._vision_processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            print(f"Vision model loaded successfully!")

        except Exception as e:
            print(f"Error loading vision model: {e}")
            raise

    def _setup_openai_client(self):
        """Sets up OpenAI client using API key from environment or user input."""
        # Load environment variables from .env file
        load_dotenv(ENV_FILE)

        # Try to get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            # Check if API key exists in common locations
            potential_paths = [
                Path.home() / '.openai' / 'api_key.txt',
                Path.home() / '.config' / 'openai' / 'api_key.txt'
            ]

            for path in potential_paths:
                if path.exists():
                    try:
                        api_key = path.read_text().strip()
                        print(f"Found API key at {path}")
                        break
                    except:
                        pass

        # If still no API key, ask for input
        if not api_key:
            print("OpenAI API key not found in environment or config files.")
            print("Please enter your OpenAI API key:")
            api_key = input().strip()

            # Save to environment variable for future use in the session
            os.environ['OPENAI_API_KEY'] = api_key

            # Ask if user wants to save the key
            save_key = input(
                "Save API key for future sessions? (y/n): ").lower()
            if save_key == 'y':
                config_dir = Path.home() / '.config' / 'openai'
                config_dir.mkdir(parents=True, exist_ok=True)
                key_path = config_dir / 'api_key.txt'
                key_path.write_text(api_key)
                print(f"API key saved to {key_path}")

        try:
            self._openai_client = OpenAI(api_key=api_key)
            # Test the client with a simple request
            self._openai_client.models.list(limit=1)
            print("OpenAI client configured successfully!")

        except Exception as e:
            print(f"Error setting up OpenAI client: {e}")
            raise


# Create a singleton instance for global use
model_manager = ModelManager()
