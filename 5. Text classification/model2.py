import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

# Load training and test data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform tags in training data
train_df['encoded_tags'] = label_encoder.fit_transform(train_df['tag'].astype(str))

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained('monsoon-nlp/bert-base-thai')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# Tokenize training and test data
train_encodings = tokenize_function(train_df['description'].astype(str).tolist())
test_encodings = tokenize_function(test_df['description'].astype(str).tolist())

# Prepare dataset for PyTorch
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings['input_ids'])

# Create datasets
train_dataset = CustomDataset(train_encodings, train_df['encoded_tags'].tolist())
test_dataset = CustomDataset(test_encodings)

# Load model
model = AutoModelForSequenceClassification.from_pretrained('monsoon-nlp/bert-base-thai', num_labels=len(label_encoder.classes_))

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Predict
predictions = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# Convert predicted labels back to original tags
test_df['predicted_tag'] = label_encoder.inverse_transform(predicted_labels)

# Prepare output
output_df = test_df[['index', 'predicted_tag']]
output_df.to_csv('output.csv', index=False)

print("Predictions saved to 'submission.csv'.")
