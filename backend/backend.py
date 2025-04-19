
from fastapi.concurrency import asynccontextmanager
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

import lightning.pytorch as pl
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torchmetrics import Accuracy

import os
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification


# Set seed for reproducibility
pl.seed_everything(42)

torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = False  # For deterministic results
torch.backends.cudnn.benchmark = True  # Disabling to ensure deterministic algorithm
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # Optimize matmul precision

# Cell 2: Data Module and Dataset Definition

class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DepressionDataModule(pl.LightningDataModule):
    def __init__(self, df, tokenizer, max_length=128, batch_size=16):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Split the dataset into training and validation sets (80/20 split)
        train_df, val_df = train_test_split(
            self.df, test_size=0.2, random_state=42, stratify=self.df['label']
        )
        self.train_dataset = DepressionDataset(
            train_df['body'].tolist(), train_df['label'].tolist(), self.tokenizer, self.max_length
        )
        self.val_dataset = DepressionDataset(
            val_df['body'].tolist(), val_df['label'].tolist(), self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
# Cell 3: Model Definition

from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

class DepressionClassifier(pl.LightningModule):
    def __init__(self, n_classes, steps_per_epoch=None, n_epochs=None, lr=2e-5):
        super().__init__()
        # Use DistilRoBERTa for faster training
        self.model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=n_classes).train()
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        # Separate accuracy metrics for training and validation
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, batch['labels'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Compute and print training accuracy for the epoch
        train_epoch_acc = self.train_acc.compute()
        print(f"Epoch {self.current_epoch} - Training Accuracy: {train_epoch_acc:.4f}")
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, batch['labels'])
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute and print validation accuracy for the epoch
        val_epoch_acc = self.val_acc.compute()
        print(f"Epoch {self.current_epoch} - Validation Accuracy: {val_epoch_acc:.4f}")
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.steps_per_epoch is None or self.n_epochs is None:
            return optimizer
        total_steps = self.steps_per_epoch * self.n_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

# Define request model
class TextRequest(BaseModel):
    text: str = Field(..., description="The input text to classify", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text that needs to be classified."
            }
        }

# Define response model
class PredictionResponse(BaseModel):
    result: bool = Field(..., description="Classification result (True = 1, False = 0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result": True,
            }
        }

# Define lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global model
    global tokenizer
    global device
    
    # Load model - either from a checkpoint or use a pre-trained model
    checkpoint_path = "./best-checkpoint.ckpt"
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 2
    
    try:
        # if checkpoint_path and os.path.exists(checkpoint_path):
        global model
        # Load from checkpoint
        model = DepressionClassifier.load_from_checkpoint(checkpoint_path, n_classes=n_classes)
        model.eval()
        model.to(device)
        # else:
        #     print("load failed")
        # Put model in evaluation mode
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None
    
    yield  # This is where the app runs
    
    # Shutdown: Release resources
    model = None
    print("Model resources released")

# Initialize FastAPI app with updated configurations and lifespan
app = FastAPI(
    title="DeepMindCare Detection API",
    description="Detect Depression using DeepMindCare AI models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model():
    """Dependency to get the model instance"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model

@app.post("/predict", response_model=PredictionResponse, status_code=200)
async def predict(request: TextRequest, model: DepressionClassifier = Depends(get_model)) -> Dict[str, Any]:
    print(model)
    try:
        encoding = tokenizer.encode_plus(
        request.text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
    
    # Run the model in evaluation mode
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return PredictionResponse(result=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

# Example usage endpoints with sample requests
@app.get("/examples/request", response_model=TextRequest)
async def example_request():
    """Returns an example TextRequest object"""
    return TextRequest(text="This is a sample text that needs to be classified.")

@app.get("/examples/response", response_model=PredictionResponse)
async def example_response():
    """Returns an example PredictionResponse object"""
    return PredictionResponse(result=True)

if __name__ == "__main__":
    import uvicorn
    # Run the API server on port 8000
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)