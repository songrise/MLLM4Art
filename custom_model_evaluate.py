import argparse
import os
import json
from abc import ABC, abstractmethod
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import tqdm
import time


class Evaluator(ABC):
    """Abstract base class for image quality evaluators.
    Please implement the evaluate and better_than methods."""
    
    @abstractmethod
    def evaluate(self, image_path: str, prompt: str) -> float:
        """
        Evaluate an image given a prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for evaluation
            
        Returns:
            A score representing the quality/alignment
        """
        pass
    
    @abstractmethod
    def better_than(self, score1: float, score2: float) -> bool:
        """
        Compare two scores.
        
        Args:
            score1: First score
            score2: Second score
            
        Returns:
            True if score1 is better than score2
        """
        pass


class RandomGuessEvaluator(Evaluator):
    """A simple evaluator that randomly guesses scores."""
    
    def evaluate(self, image_path: str, prompt: str) -> float:
        import random
        time.sleep(0.001)  # Simulate some processing time
        return random.random()
    
    def better_than(self, score1: float, score2: float) -> bool:
        return score1 > score2

class PickScoreEvaluator(Evaluator):
    """An Exemplar CLIP-based evaluator using PickScore."""
    
    def __init__(
        self,
        processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_name: str = "yuvalkirstain/PickScore_v1",
        device: str = "cuda"
    ):
        """
        Initialize CLIP evaluator.
        
        Args:
            processor_name: Name or path of the processor
            model_name: Name or path of the pretrained model
            device: Device to run the model on
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    def calc_logits(self, prompt: str, images: List[Image.Image]) -> List[float]:
        """
        Calculate logits for images given a prompt.
        
        Args:
            prompt: Text prompt
            images: List of PIL images
            
        Returns:
            List of logit scores
        """
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            # Embed images
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
            # Embed text
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
            # Calculate scores
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        return scores.cpu().tolist()
    
    def calc_probs(self, prompt: str, images: List[Image.Image]) -> List[float]:
        """
        Calculate probabilities for images given a prompt.
        
        Args:
            prompt: Text prompt
            images: List of PIL images
            
        Returns:
            List of probability scores
        """
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            # Embed images
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
            # Embed text
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
            # Calculate scores
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # Get probabilities
            probs = torch.softmax(scores, dim=-1)
        
        return probs.cpu().tolist()
    
    def evaluate(self, image_path: str, prompt: str) -> float:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for evaluation
            
        Returns:
            Logit score for the image
        """
        image = Image.open(image_path)
        return self.calc_logits(prompt, [image])[0]
    
    def better_than(self, score1: float, score2: float) -> bool:
        """
        Compare two scores (higher is better for CLIP).
        
        Args:
            score1: First score
            score2: Second score
            
        Returns:
            True if score1 is better than score2
        """
        return score1 > score2


def evaluate_FineArtBench(
    evaluator: Evaluator,
    afc_annotation_path: str,
    combination_annotations_path: str,
    style_annotation_path: str,
    output_dir: str,
    output_annotation_path: str,
    max_samples: int = None,
) -> None:
    """
    Evaluate a dataset using the given evaluator.
    
    Args:
        evaluator: Evaluator instance to use
        afc_annotation_path: Path to 2AFC annotation file
        combination_annotations_path: Path to style combination file
        style_annotation_path: Path to style annotation file
        output_dir: Directory containing output images
        output_annotation_path: Path to save results
        max_samples: Maximum number of samples to process (None for all)

    """
    # Load annotations
    with open(style_annotation_path, "r") as f:
        style_annotation = json.load(f)
    
    with open(afc_annotation_path, "r") as f:
        annotations = json.load(f)
    
    with open(combination_annotations_path, "r") as f:
        style_combinations = json.load(f)
    
    # Process samples
    num_samples = len(annotations) if max_samples is None else min(max_samples, len(annotations))
    
    for i in tqdm.tqdm(range(num_samples)):
    
        annotation = annotations[i]
        content_index = annotation["content_index"]
        style_index = annotation["style_index"]
        
        # Get style combination
        combination_item = style_combinations[str(content_index)]
        
        # Find style offset
        style_offset = 0
        for offset, style in enumerate(combination_item):
            if style == style_index:
                style_offset = offset
                break
        style_offset += 1  # 1-indexed
        
        # Get method names
        left_name = annotation["method_left"]
        right_name = annotation["method_right"]
        
        # Construct image paths
        left_img_path = os.path.join(
            output_dir,
            left_name,
            f"content_{content_index}_style_{style_offset}.jpg"
        )
        right_img_path = os.path.join(
            output_dir,
            right_name,
            f"content_{content_index}_style_{style_offset}.jpg"
        )
        
        # Get style prompt
        style_prompt = style_annotation[str(style_index)]["prompt"]
        
        # print(
        #     f"Content index: {content_index}, Style index: {style_index}, "
        #     f"Style offset: {style_offset}, style_prompt: {style_prompt}"
        # )
        
        # Evaluate images
        left_score = evaluator.evaluate(left_img_path, style_prompt)
        right_score = evaluator.evaluate(right_img_path, style_prompt)
        
        # Determine winner
        winner = "left" if evaluator.better_than(left_score, right_score) else "right"
        
        annotations[i]["winner"] = winner
    
    # Save results
    os.makedirs(os.path.dirname(output_annotation_path), exist_ok=True)
    with open(output_annotation_path, "w") as f:
        json.dump(annotations, f, indent=4)
    
    print(f"Results saved to {output_annotation_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models on image evaluation datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', '--m',
        type = str,
        choices= ['global', 'instance'],
        default = 'global',
        help = 'Evaluation mode: global (per-artist) or instance (per-instance)'
    )
    # JSON annotation paths 
    parser.add_argument(
        "--afc-base",
        type=str,
        default="./data/2AFC/",
        help="path to 2AFC base directory"
    )
    parser.add_argument(
        "--combination-annotations-path",
        type=str,
        default="./data/painting/content_style_combination.json",
        help="Absolute path to style combination annotation file"
    )
    parser.add_argument(
        "--style-annotation-path",
        type=str,
        default="./data/base/style_1k.json",
        help="Absolute path to style annotation file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./out/",
        help="path to output annotation directory"
    )
    
    # Output images directory
    parser.add_argument(
        "--output-images-dir",
        type=str,
        default="./out",
        help="Absolute path to directory containing output images"
    )
    
    # Processing arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RandomGuessEvaluator()
    
    # Evaluate dataset
    print(f"\n{'='*60}")
    print(f"Processing dataset")
    print(f"{'='*60}\n")
    
    mode = args.mode
    print(f"Evaluation mode: {mode}")
    afc_annotation_path = os.path.join(
        args.afc_base,
        f"2AFC_{mode}_N_5000.json"
    )


    os.makedirs(args.output_dir, exist_ok=True)
    output_annotation_path = os.path.join(
        args.output_dir,
        f"random_guess_annotation_{mode}.json"
    )

    evaluate_FineArtBench(
        evaluator=evaluator,
        afc_annotation_path=afc_annotation_path,
        combination_annotations_path=args.combination_annotations_path,
        style_annotation_path=args.style_annotation_path,
        output_dir=args.output_images_dir,
        output_annotation_path=output_annotation_path,
        max_samples=args.max_samples,
    )
    
    print(f"\n{'='*60}")
    print("Processing completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()