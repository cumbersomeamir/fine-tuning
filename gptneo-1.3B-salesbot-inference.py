!pip3 install transformers
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, pipeline

# Load the tokenizer and model
model_name = "Amirkid/improving-gptneo-1.3b" 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Create a pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input text
text = "What a beautiful day in london"

# Generate a completion
result = generator(text, max_length=50, do_sample=True, temperature=0.7)

# Print the result
print(result[0]['generated_text'])
