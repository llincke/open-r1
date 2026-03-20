#!/usr/bin/env python3
"""Simple interactive chat with Qwen2.5-1.5B-Instruct model using Hugging Face Transformers.
Works on CPU (slow) or GPU (fast).
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Model path (HF repo or local directory)")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    p.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    p.add_argument("--cpu", action="store_true", help="Force CPU mode (default: auto-detect GPU)")
    args = p.parse_args()
    
    # Detect device
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {args.model_dir}")
    print(f"Using device: {device}\n")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()
    
    print("=== Qwen2.5-1.5B-Instruct Chat ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True
                )
            
            # Decode and print
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
