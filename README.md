# tic-tac-transformer

### Project Proposal

The goal of this article is to implement a transformer decoder on a toy problem so that I could gain a better understanding of decoders and hopefully you can too. I chose tic-tac-toe as the toy problem for a few reason. The prime reason being it is not the standard application of transformers, and I wanted to see if the transformer that I implemented could learn the strategy.

In order to complete this tasks there are a few things we need to do.

1. Implement Tic-Tac-Toe
2. Collect the Data
3. Implement the Transformer
4. Run experiments to see if it works

If you want to skip the article and read the code yourself, here is the [github repo](https://github.com/mullisd1/tic-tac-transformer)

## Tic Tac Toe Game

The first thing we need to do is to implement the tic-tac-toe game. This is a relatively simple affair with only a little bit of numpy. The repo for the game is [here](https://github.com/mullisd1/tic-tac-toe). The repo consists of 1 class that holds the boards state as well as some test to make sure the game works as intended. The repo also allows for game boards of greater than 3x3 in size if you wanted to try to extend the work that I did here to larger boards.

## Data
Now that the game is implemented, we have to define the problem space. This just means we have to define what our model inputs and outputs will be. In this case we have a 3x3 matrix where each position holds either a -1, 0, 1 (-1 = X, 0 = empty, 1 = O). In order to make this something the model understands we will transform this 3x3 matrix into a 1x9 matrix

``` python
>>> arr
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
>>> arr.flatten()
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

The current board state will serve as the model input, and the optimal next board state will serve as the model output. In order to find the optimal next board state we will use the Min Max algorithm

### Min Max Algorithm

A Minimax tree for Tic-Tac-Toe represents all possible game states, where each node is a board configuration. The tree alternates between maximizing (AI’s turn) and minimizing (opponent’s turn) players. The AI assigns scores to terminal nodes (win = 1, loss = -1, draw = 0) and propagates these values upward, assuming the opponent plays optimally. At each decision point, the AI selects the move leading to the best possible outcome. This ensures the AI plays optimally, making it either win or force a draw in an ideal game. If you would like a better more in depth explanation you can find one [here](https://philippmuens.com/minimax-and-mcts). 

Below is an example of how a min max tree works alternating between minimizing each layer (represents our opponents turn) and maximizing each layer (represents our turn)

<div style="text-align:center"><img src="https://philippmuens.com/assets/blog/minimax-and-mcts/chess-minimax-6.png" /></div>

This does come with one complication. In Tic-Tac-Toe there are often multiple moves that hold the same value. To deal with this we simply take the move that wins the quickest.

## Model Architechture

Below is the transformer model architecture from the now famous ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper. For the purpose of this article, we will be focusing on the right block of the graph which is known as the Decoder block. The left block is what is known as the Encoder block.

In order to explain the architecture we will use a toy problem to explain the basic control flow.

***Come Back and insert basic control flow example***

<div style="text-align:center"><img src="./references/imgs/all_you_need_is_attention_full.png" /></div>

Now that we have a basic understanding of what the figure means we can explain the control flow of how our tic-tac-toe solver will work. Below I have added an image of what our decoder architecture will more closely resemble.

<div style="text-align:center"><img src="./references/imgs/all_you_need_is_attention_decoder_block_no_cross.png" /></div>

### Multihead Attention

<div style="text-align:center"><img src="./references/imgs/all_you_need_is_attention_multihead_attention.png" /></div>

The Masked Multi-Head Attention block consists of multiple attention heads stacked together. An attention head computes an attention score for each input. This is done by using ***query, key, and value*** layers. The queries and keys calculate the attention weights via a dot product. This is then scaled by square root of the dimension size. This is so the softmax function does not behave oddly when there is high dimensionality (less important for our use case). The weights are then applied to the value matrix, producing a weighted sum that highlights the relevant information.

Below the Multi-Head Attention block is also implemented. This is simply multiple attention heads running in parallel to extract different relationships from the data.

Dropout is included in the code below. Dropout is a simple process that while training the model will randomly turn off a percentage of the neurons. This allows for the model to generalize better during inference. For a better explanation look [here](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9/)

```python
class Head(nn.Module):
    """ Singular Head"""

    def __init__(self,
                 num_embedding,
                 head_size,
                 dropout
                 ):
        super().__init__()
        self.key = nn.Linear(num_embedding, head_size, bias=False)
        self.query = nn.Linear(num_embedding, head_size, bias=False)
        self.value = nn.Linear(num_embedding, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B: Batch Size
        # T: Number of Tokens
        # C: Number of channels (3 in this case for X, O, and empty)
        B,T,C = x.shape

        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Compute attention
        # C**0.5 is to make the numbers smaller so softmax doesn't do weird things
        wei = q @ k.transpose(-2, -1) / C**0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        V = self.value(x)
        out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
```
``` python
class MultiHeadAttention(nn.Module):
    """Multiple Attention Heads"""
    def __init__(self, num_heads, num_embedding, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(num_embedding, head_size, dropout) for i in range(num_heads)])
        self.project = nn.Linear(num_embedding, num_embedding)                  # Projection layer for getting back into the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.project(out)
        out = self.dropout(out)
        return out
```


### Feed Foward
<div style="text-align:center"><img src="./references/imgs/all_you_need_is_attention_feed_foward.png" /></div>

The Feed Forward block is single MLP layer with a relu activation function to add non linearity. Non-linearity is crucial because it enables the model to learn complex patterns that a purely linear system cannot capture. If the feed-forward layer were only composed of linear transformations, stacking multiple layers would still be equivalent to a single linear transformation, limiting the model’s expressiveness.

```python
class FeedForward(nn.Module):
    """Single Layer"""

    def __init__(self, num_embedding, dropout):
        super().__init__()

        self.m = nn.Sequential(nn.Linear(num_embedding, 4 * num_embedding), # 4* because they did it in the paper
                               nn.ReLU(),
                               nn.Linear(4 * num_embedding, num_embedding), # Projection layer for getting back into the residual pathway
                               nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.m(x)
```

### Decoder Block

<div style="text-align:center"><img src="./references/imgs/all_you_need_is_attention_decoder_block_no_cross.png" /></div>

Now it is time to put it is time to put it all together into a decoder block. You will notice that I also added the residual connections and layer norm to the block. Residual connections help deal with the common issue of exploding and vanishing gradients. While layer norm addresses the internal covariant shift problem.

You can find better explanations for why we use [layer norm here](https://medium.com/@sujathamudadla1213/layer-normalization-48ee115a14a4) and [residual connections here](https://medium.com/towards-data-science/what-is-residual-connection-efb07cab0d55).

```python
class Block(nn.Module):
    """"""

    def __init__(self, num_embedding, num_heads, dropout):
        super().__init__()

        head_size = num_embedding // num_heads
        self.self_attn = MultiHeadAttention(num_heads, num_embedding, head_size, dropout)
        self.feed_fwd = FeedForward(num_embedding, dropout)

        self.lay_norm1 = nn.LayerNorm(num_embedding)
        self.lay_norm2 = nn.LayerNorm(num_embedding)

    def forward(self, x):
        x = x + self.self_attn(self.lay_norm1(x))
        x = x + self.feed_fwd(self.lay_norm2(x))
        return x
```

## Model Training

Now that we can build the model, we need to be able to train it. You will notice that I added somethings that are not strictly necessary but are nice to haves.

- Early Stopping:
    - stops training before the max number of epochs is reached because the validation loss has not gone down in x# of epochs
    - stops the model training when it begins to overfit
- LR Scheduler:
    - reduces the learning rate if the validation loss has not gone down in x# of epochs
    - allows for the use of a larger learning rate in the beginning so the model learns faster

```python
class ModelTrainer:
    def __init__(self, model,
                 train_dataloader,
                 val_dataloader, 
                 lr=1e-3,
                 epochs = 100,
                 early_stopping = 100,
                 device = None):
        self.verbose = False

        self.model = model
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

        self.epochs = epochs
        self.early_stopping = early_stopping
        self.lr = lr
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss() 
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor = 0.5,
                                                                    patience = 30)  # Reduce LR when learning stops

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
            for inputs, labels in self.val_loader: #tqdm(self.val_loader, desc="Evaluating", unit="batch"):
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
        best_val_loss = np.inf
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # print(f"Epoch {epoch+1}/{self.epochs}")
            train_loss = self.train_one_epoch()
            val_loss, val_accuracy = self.evaluate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if self.verbose:
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
            
            # Save the best model based on validation accuracy
            if val_loss < best_val_loss:
                counter = 0
                best_val_loss = val_loss
                print("Saving best model...")
                torch.save(self.model.state_dict(), "./models/best_model.pth")
            else:
                counter += 1

            if counter > 50:
                print(f"Early stopping at Epoch {epoch}")
                break
        return train_losses, val_losses
```

## Experiments

Below I will list the experiments that were performed. For all of the experiments some of the config variables will remain constant.

```
vocab_size = 3 # number of tokens (X, O, empty)
block_size = 9 # board size
```

### Experiment 1: Initial Run

**Hypothesis**: To start off I trained with a relatively small model with no dropout to see if it could learn.
**Config**:
```
num_embedding = 32,
num_heads = 4,
num_blocks = 4,
dropout = 0.0
```
**Loss Graph**:
<div style="text-align:center"><img src="./references/models/exp1/losses.png" /></div>

**Test Results**: 84.459% accuracy
**Analysis**:

### Experiment 2: Add Dropout

**Hypotosis**: Dropout will help with the learning
**Config**:
```
num_embedding = 32,
num_heads = 4,
num_blocks = 4,
dropout = 0.1
```
**Loss Graph**:
<div style="text-align:center"><img src="./references/models/exp2/losses.png" /></div>

**Test Results**: 92.229% accuracy
**Analysis**:

### Experiment 3: Make the model wider

**Hypotosis**: Adding more heads will allow the model to learn better
**Config**:
```
num_embedding = 32,
num_heads = 8,
num_blocks = 4,
dropout = 0.1
```
**Loss Graph**:
<div style="text-align:center"><img src="./references/models/exp3/losses.png" /></div>

**Test Results**: 94.932% accuracy
**Analysis**:

### Experiment 4: Make the model wider

**Hypotosis**: Adding more heads will allow the model to learn better
**Config**:
```
num_embedding = 128,
num_heads = 8,
num_blocks = 4,
dropout = 0.1
```
**Loss Graph**:
<div style="text-align:center"><img src="./references/models/exp4/losses.png" /></div>

**Test Results**: 96.621% accuracy
**Analysis**:

## Conclusion:
