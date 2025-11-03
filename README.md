# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## THEORY

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load data, create word/tag mappings, and group sentences.

### STEP 2: 
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.


### STEP 3: 
Define dataset and DataLoader for batching.


### STEP 4: 
Build a bidirectional LSTM model for sequence tagging.


### STEP 5: 
Train the model over multiple epochs, tracking loss


### STEP 6: 
Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.





## PROGRAM

### Name:MARELLA HASINI

### Register Number:212223240083

```python
# Model definition
class BiLSTMTagger(nn.Module):
  def __init__(self,vocab_size,tarset_size,embedding_dim=50,hidden_dim=100):
    super(BiLSTMTagger,self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm=(nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True))
    self.fc=nn.Linear(hidden_dim*2,tarset_size)
  def forward(self,x):
    x=self.embedding(x)
    x=self.dropout(x)
    x,_=self.lstm(x)
    return self.fc(x)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)   # FIXED
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss += loss.item()  # FIXED

        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

    return train_losses, val_losses



```

### OUTPUT

## Loss Vs Epoch Plot

<img width="557" height="408" alt="Screenshot 2025-11-03 180927" src="https://github.com/user-attachments/assets/3120bd5a-8190-4ac1-b8b8-426b71ddacd7" />

### Sample Text Prediction
<img width="291" height="311" alt="image" src="https://github.com/user-attachments/assets/7d5aa955-51bd-40c7-b6b7-c24b54e4ae8e" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
