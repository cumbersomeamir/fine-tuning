#pip3 install transformers torch
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def main():
    model_name = "Amirkid/juicewrld-gptneo1.3B"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)

    print("Enter a prompt to generate text or type 'exit' to quit.")
    while True:
        prompt = input("Prompt: ")

        if prompt.lower() == "exit":
            break

        input_tokens = tokenizer.encode(prompt, return_tensors="pt")
        output_tokens = model.generate(input_tokens, max_length=100, num_return_sequences=1)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        print(f"Generated text: {output_text}")

if __name__ == "__main__":
    main()
