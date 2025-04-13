import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load data
fake = pd.read_csv(r"ByteVerse 2025\Fake News Detector\Dataset\Fake.csv")
true = pd.read_csv(r"ByteVerse 2025\Fake News Detector\Dataset\True.csv")
fake["label"] = 0
true["label"] = 1
main = pd.concat([fake, true])

indian = pd.read_csv(r"ByteVerse 2025\Fake News Detector\Dataset\indian_news.csv")
indian["label"] = indian["label"]

df = pd.concat([main, indian], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
df["combined"] = df["title"] + " " + df["text"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df["combined"], df["label"], test_size=0.2)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Custom Dataset for PyTorch
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Prepare datasets
train_dataset = FakeNewsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
test_dataset = FakeNewsDataset(X_test.tolist(), y_test.tolist(), tokenizer)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Trainer Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("fake_news_bert_model")
tokenizer.save_pretrained("fake_news_bert_model")
print("✅ Model trained and saved as fake_news_bert_model")
