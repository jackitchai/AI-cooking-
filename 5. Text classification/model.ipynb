import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preprocessing tags
train_df['tag'] = train_df['tag'].apply(lambda x: x.split('|') if pd.notnull(x) else [])

# Fill missing values in the text columns with empty strings
train_df['title'] = train_df['title'].fillna('')
train_df['description'] = train_df['description'].fillna('')
train_df['institute'] = train_df['institute'].fillna('')
test_df['title'] = test_df['title'].fillna('')
test_df['description'] = test_df['description'].fillna('')
test_df['institute'] = test_df['institute'].fillna('')

# Split data into training and validation sets
train_texts, val_texts, train_tags, val_tags = train_test_split(
    train_df[['title', 'description', 'institute']],
    train_df['tag'],
    test_size=0.2,
    random_state=42
)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize texts
def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

train_encodings = tokenize_texts(train_texts.apply(lambda x: ' '.join(x), axis=1))
val_encodings = tokenize_texts(val_texts.apply(lambda x: ' '.join(x), axis=1))
test_encodings = tokenize_texts(test_df[['title', 'description', 'institute']].apply(lambda x: ' '.join(x), axis=1))

# Binarize tags
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_tags)
val_labels = mlb.transform(val_tags)

# Create Dataset class
class CourseDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CourseDataset(train_encodings, train_labels)
val_dataset = CourseDataset(val_encodings, val_labels)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(mlb.classes_))

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Prepare test dataset
test_dataset = CourseDataset(test_encodings, torch.zeros((len(test_df), len(mlb.classes_))))

# Make predictions
predictions = trainer.predict(test_dataset)
predicted_labels = mlb.inverse_transform(predictions.predictions > 0.5)

# Create output DataFrame
output_df = pd.DataFrame({
    'index': test_df.index,
    'tag': ['|'.join(tags) for tags in predicted_labels]
})

# Save to CSV
output_df.to_csv('output.csv', index=False)
