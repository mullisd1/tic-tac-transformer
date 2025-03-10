from tqdm import tqdm

import torch
from torch import nn

class ModelTrainer:
    def __init__(self, model,
                 train_dataloader,
                 val_dataloader, 
                 lr=1e-3,
                 epochs = 100,
                 device = None):
        self.model = model
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss() 

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(self.train_loader, desc="Training", unit="batch"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()  # Zero the gradients
            logits = self.model(inputs)
            
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update model parameters
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(self.train_loader)
        return avg_train_loss
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in tqdm(self.val_loader, desc="Evaluating", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                
                outputs = torch.argmax(logits, dim=2)
                
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                running_loss += loss.item()
                
                total += labels.size(0)
                correct += (torch.all(outputs == labels, dim=1)).sum().item()
        
        avg_val_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_val_loss, accuracy
    
    def train(self):
        best_val_accuracy = 0.0
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self.train_one_epoch()
            val_loss, val_accuracy = self.evaluate()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
            
            # Save the best model based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print("Saving best model...")
                torch.save(self.model.state_dict(), "./models/best_model.pth")

    def load_best_model(self):
        self.model.load_state_dict(torch.load("./models/best_model.pth"))
        self.model.to(self.device)
        print("Best model loaded successfully.")

    def test(self, test_dataloader, model_path: str = None):
        if model_path is None:
            self.load_best_model()
        else:
            self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)

        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, labels in tqdm(test_dataloader, desc="Testing", unit="batch"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                
                
                outputs = torch.argmax(logits, dim=2)
                # print(f"outputs shape: {outputs.shape}")
                
                total += labels.size(0)
                correct += (torch.all(outputs == labels, dim=1)).sum().item()
                # input()
        
        accuracy = correct / total
        return accuracy