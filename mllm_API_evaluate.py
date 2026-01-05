# -*- coding : utf-8 -*-
# @FileName  : mllm_API_evaluate.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Dec 09, 2025
# @Github    : https://github.com/songrise
# @Description: Run ArtCoT on FineArtBench with OpenAI api.

import os
import json
import base64
from PIL import Image
import openai
from typing import List, Dict, Tuple, Optional
import ast
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import omegaconf
import pickle
import re


# === Configuration ===

# It's recommended to set your API key as an environment variable for security.
# For example, in your terminal:
# export OPENAI_API_KEY='your-api-key-here'


API_KEY = ""
BASE_URL = ""
# Initialize OpenAI with custom base URL if needed
openai.api_key = API_KEY
openai.api_base = BASE_URL

# === Function Definitions ===


def calc_stats(outputs: List[float]) -> Dict[str, float]:
    # calculate the mean and standard deviation of the scores [3, N, num_exps]
    name = ["style", "content", "aesthetic"]
    for i, output in enumerate(outputs):
        output = np.array(output)
        mean = np.mean(output, axis=0)
        std = np.std(output, axis=0)
        print(f"{name[i]}: mean = {mean}, std = {std}")


def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def VLM_infer(client, prompt: str, image_b64: str, model: str) -> Optional[str]:
    if image_b64 is not None and image_b64 != "":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                ],
            },
        ]

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract the response content
    extracted_text = response.choices[0].message.content
    return extracted_text


def parse_gpt_response(response_text: str) -> Optional[str]:
    """
    Parses the GPT model's response to extract the preference decision.

    The expected output is a Python dict that includes a 'winner' key.
    It handles various response formats, for example, responses enclosed in a Markdown
    code block (e.g., ```json or ```python), or responses that include additional text.

    The function returns:
        - "right" if the 'winner' value equals 1 or "1"
        - "left" if the 'winner' value equals 0 or "0"

    Args:
        response_text (str): The raw GPT response.

    Returns:
        Optional[str]: "right" or "left" if parsing is successful; otherwise, None.
    """

    if not response_text:
        return None

    response_text = response_text.strip()

    # Try to extract thinking and winner directly with regex for malformed inputs
    winner_pattern = re.compile(r"'winner':\s*(\d+)")
    winner_match = winner_pattern.search(response_text)
    if winner_match:
        winner_value = winner_match.group(1)
        if winner_value == "1":
            return "right"
        elif winner_value == "0":
            return "left"

    # First, try to remove Markdown code block delimiters with improved pattern
    code_block_pattern = re.compile(
        r"```(?:\w*)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
    )
    match = code_block_pattern.search(response_text)
    if match:
        cleaned_response = match.group(1).strip()
    else:
        # If no Markdown block is found, try to extract a dict-like substring
        dict_match = re.search(r"({.*})", response_text, re.DOTALL)
        if dict_match:
            cleaned_response = dict_match.group(1).strip()
        else:
            # Fallback: assume the entire response might be the decision dict
            cleaned_response = response_text

    # Handle Markdown special characters that might interfere with parsing
    markdown_chars = ['*', '_', '`', '#', '~', '>', '|', '+', '-']
    for char in markdown_chars:
        cleaned_response = cleaned_response.replace(f'\\{char}', char)
    
    # Try multiple parsing approaches
    parsed = None
    
    # 1. Direct regex approach - most resilient for malformed inputs
    winner_match = re.search(r"['\"]winner['\"]\s*:\s*([01]|['\"][01]['\"]|true|false|['\"]true['\"]|['\"]false['\"])", 
                           cleaned_response, re.IGNORECASE)
    if winner_match:
        winner_val = winner_match.group(1).lower().strip('"\'')
        if winner_val in ["1", "true"]:
            return "right"
        elif winner_val in ["0", "false"]:
            return "left"
    
    # 2. Try to fix common formatting issues before parsing
    try:
        # Replace single quotes with double quotes for JSON compatibility
        json_compatible = re.sub(r"'([^']*)':", r'"\1":', cleaned_response)
        json_compatible = re.sub(r":\s*'([^']*)'", r': "\1"', json_compatible)
        parsed = json.loads(json_compatible)
    except json.JSONDecodeError:
        try:
            # 3. Try Python literal_eval approach
            parsed = ast.literal_eval(cleaned_response)
        except (ValueError, SyntaxError):
            # 4. Last resort: manual extraction for severely malformed inputs
            thinking_pattern = re.compile(r"'thinking':\s*'(.*?)',\s*'winner'", re.DOTALL)
            thinking_match = thinking_pattern.search(response_text)
            winner_pattern = re.compile(r"'winner':\s*(\d+)\}")
            winner_match = winner_pattern.search(response_text)
            
            if winner_match:
                winner = int(winner_match.group(1))
                if winner == 1:
                    return "right"
                elif winner == 0:
                    return "left"
            return None

    # Process the successfully parsed dictionary
    if isinstance(parsed, dict) and "winner" in parsed:
        winner = parsed["winner"]
        if winner in [1, "1", True, "true", "True"]:
            return "right"
        elif winner in [0, "0", False, "false", "False"]:
            return "left"
        else:
            print(f"Invalid winner value: {winner}")
            return None
    else:
        print("Parsed data is not a valid dict with a 'winner' key.")
        return None



def dispatch(indices: List[int], n: int) -> List[List[int]]:
    """
    Splits a list of indices into n roughly equal parts.

    Args:
        indices (List[int]): The list of indices to split.
        n (int): The number of splits.

    Returns:
        List[List[int]]: A list containing n lists of indices.
    """
    k, m = divmod(len(indices), n)
    return [indices[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


# === Main Function ===


def baseline_metric(
    client,
    preference_annotation,
    selected_idx,
    output_dir,
    base_dir,
    model,
    use_zero_cot: bool = False,
    n_thread=5,
    ignore_ids=[],
    config=None,
):
    def __process_image(
        indices_slice: List[int],
        annotation: List[Dict[str, Any]],
        base_dir: str,
        prompt_template: str,
        client: Any,
        model: str,
        ignore_ids: List[int] = [],
        config=None,
    ) -> List[List[Any]]:
        """
        Processes a slice of image indices by generating prompts, making API calls, and collecting scores.

        Args:
            indices_slice (List[int]): A slice of selected indices to process.
            annotations (List[Dict[str, Any]]): Loaded annotations data.
            base_dir (str): Base directory where images are stored.
            prompts_templates (List[str]): List of prompt templates.
            num_exps (int): Number of experiments/methods.
            client (Any): The API client instance.

        Returns:
            List[List[Any]]: A list containing three lists of scores corresponding to each prompt type.
        """
        local_vlm_preference = {}
        all_conversation = {}
        current_conversation = ""
        for i in indices_slice:
            if i in ignore_ids:
                print(f"Skipping idx {i} due to ignore_ids.")
                continue
            style = annotation[i - 1].get("style_prompt", "")
            style_image_path = os.path.join(base_dir, f"{i}.jpg")

            if not os.path.isfile(style_image_path):
                print(f"Image file not found: {style_image_path}. Skipping...")
                continue

            # Encode the image to base64 once per image
            print(f"Processing Image {style_image_path}...")
            try:
                image_b64 = encode_image_to_base64(style_image_path)
            except Exception as e:
                print(f"Failed to encode image {style_image_path}: {e}. Skipping...")
                continue

            # Format the prompt
            prompt = prompt_template.format(style=style)

            # Make the API call
            try:
                response_text = VLM_infer(client, prompt, image_b64, model)
            except Exception as e:
                response_text = None
                print(f"API call failed for idx {i}: {e}. Skipping...")
                continue
            # escape unicode
            response_text = re.sub(
                r"\\u[0-9a-fA-F]{4}",
                lambda x: x.group(0).encode().decode("unicode_escape"),
                response_text,
            )
            if response_text:
                all_conversation[i] = response_text
                winner = parse_gpt_response(response_text)
                if winner:
                    local_vlm_preference[i] = winner
                    print(f"Processed idx {i}: winner = {winner}")
                else:
                    print(
                        f"Unexpected or invalid score format for Image {i}: {response_text}"
                    )
            else:
                print(f"No response received for Image {i}.")
        return local_vlm_preference, all_conversation

    vlm_preference = {
        idx + 1: "" for idx, annotation in enumerate(preference_annotation)
    }

    vlm_responses = {
        idx + 1: "" for idx, annotation in enumerate(preference_annotation)
    }

    # === Define Prompt Templates ===
    #!hardcoded 
    base_prompt_template = (
        "You are an expert in fine art. A source image (top) and two different stylized images (bottom) in the style of `{style}`  are presented to you. "
        ". Consider both the content and style, which stylized image is better in terms of overall aesthetic quality as an artwork?"
        "Return your decision in a Python Dict, ['winner':int]. `0` means the left is better while `1` means the right is better. Do not include any other string in your response."
    )
    zeroCoT_prompt_template = """{{"request": "A source image (top) and two different stylized images (bottom) in the style of `{style}` are presented to you. Consider both the content and style,  which stylized image is better in terms of overall aesthetic quality as an artwork?". Return the rationale in concrete language and your decision in short in format of a Python Dict {{ 'thinking':str, 'winner':int}}. `0` means the left is better while `1` means the right is better.",

    "response": "{{ {{'thinking': ' Let's' think step by step in concrete and objective tone,
    """
    actual_prompt_template = (
        zeroCoT_prompt_template if use_zero_cot else base_prompt_template
    )
    # === Split selected_idx into N slices ===
    slices = dispatch(selected_idx, n_thread)

    # === Execute Concurrent Processing ===
    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        # Prepare arguments for each thread
        futures = [
            executor.submit(
                __process_image,
                slice_,
                preference_annotation,
                base_dir,
                actual_prompt_template,
                client,
                model,
                ignore_ids,
                config,
            )
            for slice_ in slices
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                slice_preference_out, slice_response = future.result()
                if slice_preference_out:
                    for idx, winner in slice_preference_out.items():
                        vlm_preference[idx] = winner
                if slice_response:
                    for idx, conversation in slice_response.items():
                        vlm_responses[idx] = conversation
            except Exception as e:
                print(f"An error occurred during processing: {e}")

    return vlm_preference, vlm_responses


def CoT_metric(
    client,
    preference_annotation,
    selected_idx,
    output_dir,
    base_dir,
    model,
    n_thread=5,
    ignore_ids=[],
):

    def __process_image(
        indices_slice: List[int],
        annotation: List[Dict[str, Any]],
        base_dir: str,
        prompt_dict: Dict,
        client: Any,
        model: str,
        ignore_ids: List[int] = [],
    ) -> List[List[Any]]:
        """
        Processes a slice of image indices by generating prompts, making API calls, and collecting scores.

        Args:
            indices_slice (List[int]): A slice of selected indices to process.
            annotations (List[Dict[str, Any]]): Loaded annotations data.
            base_dir (str): Base directory where images are stored.
            prompts_templates (List[str]): List of prompt templates.
            num_exps (int): Number of experiments/methods.
            client (Any): The API client instance.

        Returns:
            List[List[Any]]: A list containing three lists of scores corresponding to each prompt type.
        """
        local_vlm_preference = {}
        all_conversation = {}
        current_conversation = ""
        for i in indices_slice:
            if i in ignore_ids:
                print(f"Skipping idx {i} due to ignore_ids.")
                continue
            style = annotation[i - 1].get("style_prompt", "")
            style_image_path = os.path.join(base_dir, f"{i}.jpg")

            if not os.path.isfile(style_image_path):
                print(f"Image file not found: {style_image_path}. Skipping...")
                continue

            # Encode the image to base64 once per image
            print(f"Processing Image {style_image_path}...")
            try:
                image_b64 = encode_image_to_base64(style_image_path)
            except Exception as e:
                print(f"Failed to encode image {style_image_path}: {e}. Skipping...")
                continue

            # Format the prompt
            cs_prompt = prompt_dict["content_style"].format(style=style)
            # prompt = prompt_template.format(style=style)
            critique_prompt = prompt_dict["critique"].format(style=style)
            summarizer_prompt = prompt_dict["summarizer"].format(style=style)
            # Make the API call
            try:
                cs_response = VLM_infer(client, cs_prompt, image_b64, model)
                current_conversation = cs_prompt
            except Exception as e:
                print(f"API call failed for idx {i}: {e}. Skipping...")
                continue
            current_conversation += cs_response

            try:
                current_conversation += critique_prompt
                critique_response = VLM_infer(client, critique_prompt, image_b64, model)
            except Exception as e:
                print(f"API call failed for idx {i}: {e}. Skipping...")
                continue
            current_conversation += critique_response

            try:
                current_conversation += summarizer_prompt
                summarizer_response = VLM_infer(
                    client, summarizer_prompt, image_b64, model
                )
            except Exception as e:
                print(f"API call failed for idx {i}: {e}. Skipping...")
                continue
            current_conversation += summarizer_response

            if summarizer_response:
                winner = parse_gpt_response(summarizer_response)
                if winner:
                    local_vlm_preference[i] = winner
                    print(f"Processed idx {i}: winner = {winner}")
                else:
                    print(
                        f"Unexpected or invalid score format for Image {i}: {response_text}"
                    )
            else:
                print(f"No response received for Image {i}.")
            all_conversation[i] = current_conversation

        return local_vlm_preference, all_conversation

    vlm_preference = {
        idx + 1: "" for idx, annotation in enumerate(preference_annotation)
    }

    vlm_responses = {
        idx + 1: "" for idx, annotation in enumerate(preference_annotation)
    }

    # === Define Prompt Templates ===
    content_style_prompt_template = (
        "You are an expert in fine art. A source image (top) Two stylized images (bottom left and bottom right) in the style of `{style}` are presented to you. "
        "Compare the content preservation and style fidelity of the two images, which one is better. "
        "Return your answer in a Python Dict, ['style_reason':str, 'content_reason':str, 'style_winner':int, 'content_winner':int]. `0` means the left is better while `1` means the right is better. Do not include any other string in your response."
    )

    style_explanation_prompt_template = (
        "You are an expert in fine art. What visual features is essential in the style of `{style}`? How should content be painted to match the style of `{style}`? "
        "Return your answer in a Python Dict, ['style_component':str]. do not include any other string in your response."
    )

    critique_prompt_template = (
        "Take a closer look at the two stylized images at the bottom in the style of `{style}`. As an expert in art, do you agree with above analysis? Compare and consider the following questions."
        "What visual features is essential for the style of `{style}`? Is the content on top well-preserved in the specific art style? Is there any artifact, distortion or inharmonious color patterns in either painting? "
        "Return your answer in a Python Dict, [reflection':str]."
    )

    summarizer_prompt_template = (
        "Now we summarize. Based on above analysis and reflection, which stylized image at the bottom is better in terms of overall aesthetic quality as an **painting of the original content (top) in another style**? Return your answer in a Python Dict, ['winner':int]. "
        "`0` means the left is better while `1` means the right is better. Do not include any other string in your response."
    )
    # === Split selected_idx into N slices ===
    slices = dispatch(selected_idx, n_thread)

    prompt_template_dict = {
        "content_style": content_style_prompt_template,
        "style_explanation": style_explanation_prompt_template,
        "critique": critique_prompt_template,
        "summarizer": summarizer_prompt_template,
    }

    # === Execute Concurrent Processing ===
    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        # Prepare arguments for each thread
        futures = [
            executor.submit(
                __process_image,
                slice_,
                preference_annotation,
                base_dir,
                prompt_template_dict,
                client,
                model,
                ignore_ids,
            )
            for slice_ in slices
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                slice_preference_out, slice_conversation = future.result()
                if slice_preference_out:
                    for idx, winner in slice_preference_out.items():
                        vlm_preference[idx] = winner
                if slice_conversation:
                    for idx, conversation in slice_conversation.items():
                        vlm_responses[idx] = conversation

            except Exception as e:
                print(f"An error occurred during processing: {e}")

    return vlm_preference, vlm_responses


def main():
    # === Paths Configuration ===
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", type=str, default="APIConfig/gemini.yaml"
    )
    args.add_argument("--n_thread", help="Number of threads for concurrent API query", type=int, default=4)
    # args.add_argument("--sub_sample",help="subsample " type=int, default=1)
    # args.add_argument("--no_content", type=bool, default=False)
    # args.add_argument("--no_style", type=bool, default=False)
    args.add_argument("--resume", help="when set as true, will try to resume from previous execution", type=bool, default=True)
    args = args.parse_args()
    try_resume = args.resume
    config = omegaconf.OmegaConf.load(args.config)
    prefer_annotation = config.twoafc_annotation
    output_dir = config.output_dir
    dataset_annotation = config.dataset_annotation
    subset_before = config.subset_before_split # e.g., 0.2 means use first 20% of the data
    afc_base_dir = config.twoafc_base_dir
    exp_name = config.exp_name
    prompting_method = config.prompting_method
    try:
        ignore_ids = pickle.load(open(config.ignore_ids, "rb"))
    except:
        ignore_ids = []

    client = openai.OpenAI(
        api_key=API_KEY, base_url=BASE_URL
    )  # Ensure API_KEY and BASE_URL are defined
    attempt_output_dir = f"{output_dir}/{exp_name}_out.json"
    if try_resume and os.path.exists(attempt_output_dir):
        print(f"resume from {attempt_output_dir}")
        prefer_annotation = json.load(open(attempt_output_dir, "r"))
        #skip if not have the conversation
        if "conversation" in prefer_annotation[0]:
            for i in range(len(prefer_annotation)):
                if prefer_annotation[i]["winner"] != "":# and prefer_annotation[i]["conversation"] != "":
                    #copy the conversation
                    ignore_ids.append(prefer_annotation[i]["id"])
        else:
            for i in range(len(prefer_annotation)):
                if prefer_annotation[i]["winner"] != "" :
                    ignore_ids.append(prefer_annotation[i]["id"])
        attempt_output_dir = f"{output_dir}/{exp_name}_out_resumed.json"
    # === Load Annotations ===
    else:
        with open(prefer_annotation, "r", encoding="utf-8") as f:
            prefer_annotation = json.load(f)

    print(f"Skip {len(ignore_ids)} images")
    len_2afc = len(prefer_annotation)
    selected_idx = list(range(1, int(subset_before * len_2afc)))
    hello = f"hello, are you {config.model}?"
    response = VLM_infer(client, hello, "", model=config.model)
    print(response)

    if prompting_method == "base":
        vlm_preference, vlm_responses = baseline_metric(
            client,
            prefer_annotation,
            selected_idx,
            output_dir,
            afc_base_dir,
            use_zero_cot=False,
            model=config.model,
            n_thread=args.n_thread,
            ignore_ids=ignore_ids,
            config=config,
        )
    elif prompting_method == "zero_cot":
        vlm_preference, vlm_responses = baseline_metric(
            client,
            prefer_annotation,
            selected_idx,
            output_dir,
            afc_base_dir,
            use_zero_cot=True,
            model=config.model,
            n_thread=args.n_thread,
            ignore_ids=ignore_ids,
            config=config,
        )
    elif prompting_method == "art_cot":
        vlm_preference, vlm_responses = CoT_metric(
            client,
            prefer_annotation,
            selected_idx,
            output_dir,
            afc_base_dir,
            model=config.model,
            n_thread=args.n_thread,
            ignore_ids=ignore_ids,
        )

    # write to preference annotation
    for i in range(len(prefer_annotation)):
        idx = prefer_annotation[i]["id"]
        if vlm_preference[idx] != "":
            prefer_annotation[i]["winner"] = vlm_preference[idx]
        # if prompting_method == "art_cot":
        prefer_annotation[i]["conversation"] = vlm_responses[idx]

    # === Save the Outputs ===
    # export_json(selected_idx, outputs, output_dir, output_prefix)
    json.dump(prefer_annotation, open(attempt_output_dir, "w"))
    # # === Calculate Statistics ===
    # calc_stats(outputs)
    print(f"Saved to {attempt_output_dir}")


if __name__ == "__main__":
    main()
