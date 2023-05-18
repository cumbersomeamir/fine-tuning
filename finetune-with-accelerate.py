#pip3 install transformers pandas openpyxl datasets accelerate tqdm

#Importing the relevant libararies
import pandas as pd
from datasets import Dataset
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TrainingArguments, Trainer , AdamW , get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm


#Defining the accelerator object
accelerator = Accelerator()


#Reading the file
data = pd.read_excel("Juice Wrld small dataset.xlsx")

# Convert the pandas DataFrame to Hugging Face's Dataset
hf_dataset = Dataset.from_pandas(data)

# Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer.pad_token = tokenizer.eos_token



#Tokenisation
def tokenize_function(examples):
    tokenized_prompt = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    tokenized_completion = tokenizer(examples['completion'], truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    return {'input_ids': tokenized_prompt['input_ids'], 'attention_mask': tokenized_prompt['attention_mask'], 'labels': tokenized_completion['input_ids']}

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
print(tokenized_dataset)

#Removing the string columns 
column_names = ["prompt", "completion"]
tokenized_datasets = tokenized_datasets.remove_columns(dataset_dict["train"].column_names)


# Load the pre-trained GPT-Neo 1.3B model
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')

optimizer = AdamW(model.parameters(), lr=3e-5)


num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
 )


# Define the training arguments
training_args = TrainingArguments(
    output_dir='output',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_total_limit=1,
    logging_steps=100,
    evaluation_strategy="no",
)

progress_bar = tqdm(range(num_training_steps))

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
model.train()

for epoch in range(num_epochs):
      for batch in train_dataloader:
        
          outputs = model(**batch)
          loss = outputs.loss
          
          accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)

# Save the fine-tuned model
model.save_model('fine_tuned_gpt_neo_1.3B')

model = GPTNeoForCausalLM.from_pretrained('fine_tuned_gpt_neo_1.3B')


#Saving the Model on huggingface
token = "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"
model.push_to_hub("Amirkid/juicewrld-gptneo1.3B", use_auth_token=token)
