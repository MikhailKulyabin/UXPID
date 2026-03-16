import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ForumDataset(Dataset):
    """Dataset class for forum classification (both multi-label and single-label)."""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512, is_multilabel: bool = True):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: Encoded labels array (multi-hot for multi-label, class indices for single-label)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            is_multilabel: Whether this is multi-label or single-label classification
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_multilabel = is_multilabel
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        if self.is_multilabel:
            # Multi-label: return as float tensor for BCEWithLogitsLoss
            labels = torch.FloatTensor(self.labels[idx])
        else:
            # Single-label: return as long tensor for CrossEntropyLoss
            labels = torch.LongTensor([self.labels[idx]])[0]  # Get the class index
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }


class BERTMultiLabelClassifier(nn.Module):
    """BERT-based classifier for both multi-label and single-label classification."""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.3):
        """
        Initialize the BERT classifier.
        
        Args:
            model_name: Name of the pre-trained BERT model
            num_labels: Number of output labels
            dropout_rate: Dropout rate for regularization
        """
        super(BERTMultiLabelClassifier, self).__init__()
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Model logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # BERT-style models with pooler
            pooled_output = outputs.pooler_output
        else:
            # DistilBERT-style models without pooler
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]
        
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits


class BERTTrainer:
    """Trainer class for BERT classification (both multi-label and single-label)."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512,
                 batch_size: int = 16, learning_rate: float = 2e-5, 
                 num_epochs: int = 3, warmup_steps: int = 500, 
                 resume_from_checkpoint: str = None, models_output_dir: str = None,
                 text_field: str = "text", weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0, dropout_rate: float = 0.2,
                 use_class_weights: bool = False, class_weights: np.ndarray = None,
                 target_field: str = "topics"):
        """
        Initialize the trainer.
        
        Args:
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            resume_from_checkpoint: Path to checkpoint to resume training from
            models_output_dir: Directory to save model files
            text_field: Field to use for training: "text" or "insight_summary"
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
            dropout_rate: Dropout rate for model regularization
            use_class_weights: Whether to use class weights for loss calculation
            class_weights: Pre-calculated class weights array
            target_field: Target field for classification (topics, branch_status, etc.)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.models_output_dir = models_output_dir or "."
        self.text_field = text_field
        self.target_field = target_field
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.dropout_rate = dropout_rate
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        
        # Determine if this is multi-label or single-label classification
        self.is_multilabel = (target_field == "topics")
        
        # Ensure models output directory exists
        os.makedirs(self.models_output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.mlb = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def _get_loss_function(self):
        """
        Get the appropriate loss function based on classification type and class weighting configuration.
        
        Returns:
            Loss function (either weighted or standard loss)
        """
        if self.is_multilabel:
            # Multi-label classification: BCEWithLogitsLoss
            if self.use_class_weights and self.class_weights is not None:
                # Convert class weights to tensor and move to device
                pos_weights = torch.FloatTensor(self.class_weights).to(self.device)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            else:
                return nn.BCEWithLogitsLoss()
        else:
            # Single-label classification: CrossEntropyLoss
            if self.use_class_weights and self.class_weights is not None:
                # For CrossEntropyLoss, class weights are different from pos_weights
                class_weights_tensor = torch.FloatTensor(self.class_weights).to(self.device)
                return nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                return nn.CrossEntropyLoss()
    
    def set_class_weights(self, class_weights: np.ndarray):
        """
        Set class weights for weighted loss calculation.
        
        Args:
            class_weights: Array of class weights
        """
        self.class_weights = class_weights
        self.use_class_weights = True if class_weights is not None else False
        
        if self.use_class_weights:
            print(f"Class weights set: {self.class_weights}")
        else:
            print("Class weights disabled")
    
    def load_data(self, data_dir: str = "processed_data") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed training and test data with target field suffix.
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Load shared train/test files
        train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
        
        # Convert all label columns back to lists (they were saved as strings)
        import ast
        label_columns_to_convert = ['topics_labels', 'branch_status_labels', 'branch_type_labels', 'sentiment_labels']
        
        for col in label_columns_to_convert:
            if col in train_df.columns:
                train_df[col] = train_df[col].apply(ast.literal_eval)
            if col in test_df.columns:
                test_df[col] = test_df[col].apply(ast.literal_eval)
        
        # Load target-specific class info and create fresh MultiLabelBinarizer
        target_suffix = f"_{self.target_field}"
        class_info_path = os.path.join(data_dir, f"class_info{target_suffix}.json")
        with open(class_info_path, "r") as f:
            class_info = json.load(f)
        
        # Create fresh MultiLabelBinarizer to avoid version compatibility issues
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([class_info["classes"]])
        
        print(f"Loaded training data for target '{self.target_field}': {len(train_df)} samples")
        print(f"Loaded test data for target '{self.target_field}': {len(test_df)} samples")
        print(f"Number of labels: {len(self.mlb.classes_)}")
        
        return train_df, test_df
    
    def prepare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                        validation_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare datasets and data loaders.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Validate that the specified text field exists
        if self.text_field not in train_df.columns:
            raise ValueError(f"Text field '{self.text_field}' not found in training data. Available columns: {list(train_df.columns)}")
        if self.text_field not in test_df.columns:
            raise ValueError(f"Text field '{self.text_field}' not found in test data. Available columns: {list(test_df.columns)}")
        
        print(f"Using '{self.text_field}' field for training and evaluation")
        
        # Get the correct labels column based on target field
        if self.target_field == "topics":
            labels_column = 'topics_labels'
        elif self.target_field == "branch_status":
            labels_column = 'branch_status_labels'
        elif self.target_field == "branch_type":
            labels_column = 'branch_type_labels'
        elif self.target_field == "overall_branch_sentiment":
            labels_column = 'sentiment_labels'
        else:
            raise ValueError(f"Unknown target field: {self.target_field}")
        
        # Check if the labels column exists
        if labels_column not in train_df.columns:
            raise ValueError(f"Labels column '{labels_column}' not found in training data. Available columns: {list(train_df.columns)}")
        
        # Prepare training data using the specified text field
        train_texts = train_df[self.text_field].tolist()
        
        # Encode labels based on classification type and available columns
        if self.is_multilabel and self.target_field == "topics":
            # Check if one-hot encoded topic columns exist
            topic_columns = [col for col in train_df.columns if col.startswith('topic_id_')]
            
            if topic_columns:
                print(f"Using one-hot encoded topic columns: {len(topic_columns)} columns found")
                # Use one-hot encoded columns directly
                train_labels = train_df[topic_columns].values.astype(float)
                
                # Update the MultiLabelBinarizer classes to match the column order
                topic_ids = [col.replace('topic_id_', '') for col in topic_columns]
                self.mlb.classes_ = np.array(topic_ids)
                print(f"Topic classes from one-hot columns: {list(self.mlb.classes_)}")
            else:
                print("One-hot encoded columns not found, using list-based labels")
                # Fall back to list-based approach
                train_label_lists = train_df[labels_column].tolist()
                train_labels = self.mlb.transform(train_label_lists).astype(float)
        elif self.is_multilabel:
            # Multi-label for non-topics (shouldn't happen currently, but for completeness)
            train_label_lists = train_df[labels_column].tolist()
            train_labels = self.mlb.transform(train_label_lists).astype(float)
        else:
            # Single-label: convert to class indices
            train_label_lists = train_df[labels_column].tolist()
            # For single-label, each sample should have exactly one label
            train_labels = []
            for i, label_list in enumerate(train_label_lists):
                if len(label_list) == 0:
                    print(f"Warning: Sample {i} has empty labels: {label_list}. Skipping this sample.")
                    continue
                elif len(label_list) != 1:
                    print(f"Warning: Sample {i} has multiple labels for single-label classification: {label_list}. Using first label.")
                    # Take the first label if multiple labels exist
                    label = label_list[0]
                else:
                    label = label_list[0]
                
                # Get the index of this label in the classes
                if label in self.mlb.classes_:
                    label_idx = list(self.mlb.classes_).index(label)
                    train_labels.append(label_idx)
                else:
                    print(f"Warning: Label '{label}' not found in classes {list(self.mlb.classes_)}. Skipping sample {i}.")
                    continue
            
            train_labels = np.array(train_labels)
            
            # Filter training texts to match the valid labels
            if len(train_labels) < len(train_texts):
                print(f"Filtered {len(train_texts) - len(train_labels)} samples with invalid labels")
                # This is tricky - we need to keep track of which samples were valid
                # For now, let's rebuild both lists together
                valid_texts = []
                valid_labels = []
                for i, label_list in enumerate(train_label_lists):
                    if len(label_list) >= 1:
                        label = label_list[0]
                        if label in self.mlb.classes_:
                            valid_texts.append(train_texts[i])
                            label_idx = list(self.mlb.classes_).index(label)
                            valid_labels.append(label_idx)
                
                train_texts = valid_texts
                train_labels = np.array(valid_labels)
        
        # Split training data into train and validation
        val_size = int(len(train_texts) * validation_split)
        val_texts = train_texts[:val_size]
        val_labels = train_labels[:val_size]
        train_texts = train_texts[val_size:]
        train_labels = train_labels[val_size:]
        
        # Prepare test data using the specified text field
        test_texts = test_df[self.text_field].tolist()
        
        # Encode test labels using same method as training
        if self.is_multilabel and self.target_field == "topics":
            # Check if one-hot encoded topic columns exist
            topic_columns = [col for col in test_df.columns if col.startswith('topic_id_')]
            
            if topic_columns:
                print(f"Using one-hot encoded topic columns for test data: {len(topic_columns)} columns found")
                # Use one-hot encoded columns directly
                test_labels = test_df[topic_columns].values.astype(float)
            else:
                print("One-hot encoded columns not found in test data, using list-based labels")
                # Check if the labels column exists in test data
                if labels_column not in test_df.columns:
                    raise ValueError(f"Labels column '{labels_column}' not found in test data. Available columns: {list(test_df.columns)}")
                
                # Fall back to list-based approach
                test_label_lists = test_df[labels_column].tolist()
                test_labels = self.mlb.transform(test_label_lists).astype(float)
        elif self.is_multilabel:
            # Multi-label for non-topics
            # Check if the labels column exists in test data
            if labels_column not in test_df.columns:
                raise ValueError(f"Labels column '{labels_column}' not found in test data. Available columns: {list(test_df.columns)}")
            
            test_label_lists = test_df[labels_column].tolist()
            test_labels = self.mlb.transform(test_label_lists).astype(float)
        else:
            # Single-label: convert to class indices
            # Check if the labels column exists in test data
            if labels_column not in test_df.columns:
                raise ValueError(f"Labels column '{labels_column}' not found in test data. Available columns: {list(test_df.columns)}")
            
            # Get test labels from the pre-computed labels column
            test_label_lists = test_df[labels_column].tolist()
            
            valid_test_texts = []
            valid_test_labels = []
            for i, label_list in enumerate(test_label_lists):
                if len(label_list) >= 1:
                    label = label_list[0]
                    if label in self.mlb.classes_:
                        valid_test_texts.append(test_texts[i])
                        label_idx = list(self.mlb.classes_).index(label)
                        valid_test_labels.append(label_idx)
                    else:
                        print(f"Warning: Test label '{label}' not found in classes. Skipping sample {i}.")
                else:
                    print(f"Warning: Test sample {i} has empty labels. Skipping.")
            
            test_texts = valid_test_texts
            test_labels = np.array(valid_test_labels)
            
            if len(test_labels) < len(test_label_lists):
                print(f"Filtered {len(test_label_lists) - len(test_labels)} test samples with invalid labels")
        
        # Create datasets
        train_dataset = ForumDataset(train_texts, train_labels, self.tokenizer, self.max_length, self.is_multilabel)
        val_dataset = ForumDataset(val_texts, val_labels, self.tokenizer, self.max_length, self.is_multilabel)
        test_dataset = ForumDataset(test_texts, test_labels, self.tokenizer, self.max_length, self.is_multilabel)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self, num_labels: int):
        """
        Initialize the BERT model and optimizer.
        
        Args:
            num_labels: Number of output labels
        """
        # Check if we should resume from checkpoint
        if self.resume_from_checkpoint and os.path.exists(self.resume_from_checkpoint):
            self.load_from_checkpoint(self.resume_from_checkpoint)
        else:
            # Initialize fresh model
            self.model = BERTMultiLabelClassifier(self.model_name, num_labels, dropout_rate=self.dropout_rate)
            self.model.to(self.device)
            
            # Initialize optimizer with weight decay
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            print(f"Model initialized with {num_labels} labels")
            print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss (Binary Cross Entropy for multi-label with optional class weights)
            loss_fn = self._get_loss_function()
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Store predictions for metrics
            if self.is_multilabel:
                # Multi-label: use sigmoid to get probabilities for each class
                predictions = torch.sigmoid(logits).detach().cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
            else:
                # Single-label: use softmax to get class probabilities, then get predicted class
                predictions = torch.softmax(logits, dim=1).detach().cpu().numpy()
                predicted_classes = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_predictions.extend(predicted_classes)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self.calculate_metrics(np.array(all_predictions), np.array(all_labels))
        
        return avg_loss, metrics
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss with optional class weights
                loss_fn = self._get_loss_function()
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
                
                # Store predictions for metrics
                if self.is_multilabel:
                    # Multi-label: use sigmoid to get probabilities for each class
                    predictions = torch.sigmoid(logits).detach().cpu().numpy()
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
                else:
                    # Single-label: use softmax to get class probabilities, then get predicted class
                    predictions = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    predicted_classes = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    all_predictions.extend(predicted_classes)
                    all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        metrics = self.calculate_metrics(np.array(all_predictions), np.array(all_labels))
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate classification metrics for both multi-label and single-label cases.
        
        Args:
            predictions: Model predictions (probabilities for multi-label, class indices for single-label)
            labels: True labels (multi-hot for multi-label, class indices for single-label)
            threshold: Threshold for binary predictions (only used for multi-label)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if self.is_multilabel:
            # Multi-label classification metrics
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > threshold).astype(int)
            
            # Check for over-prediction (early training issue)
            avg_predictions_per_sample = binary_predictions.sum(axis=1).mean()
            avg_true_labels_per_sample = labels.sum(axis=1).mean()
            
            # If we're predicting way too many labels, try a higher threshold
            if avg_predictions_per_sample > avg_true_labels_per_sample * 3:
                # Adaptively increase threshold
                adaptive_threshold = min(0.8, threshold + 0.2)
                binary_predictions = (predictions > adaptive_threshold).astype(int)
                metrics['threshold_used'] = adaptive_threshold
                metrics['avg_predictions_per_sample'] = binary_predictions.sum(axis=1).mean()
            else:
                metrics['threshold_used'] = threshold
                metrics['avg_predictions_per_sample'] = avg_predictions_per_sample
            
            metrics['avg_true_labels_per_sample'] = avg_true_labels_per_sample
            
            # Subset accuracy (exact match)
            metrics['subset_accuracy'] = accuracy_score(labels, binary_predictions)
            
            # Hamming loss
            metrics['hamming_loss'] = hamming_loss(labels, binary_predictions)
            
            # Jaccard score (IoU) - handle case where no predictions are made
            try:
                metrics['jaccard_score'] = jaccard_score(labels, binary_predictions, average='samples')
            except:
                metrics['jaccard_score'] = 0.0
            
            # Precision, Recall, F1 (macro and micro averages)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                labels, binary_predictions, average='macro', zero_division=0
            )
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                labels, binary_predictions, average='micro', zero_division=0
            )
            
            metrics['precision_macro'] = precision_macro
            metrics['recall_macro'] = recall_macro
            metrics['f1_macro'] = f1_macro
            metrics['precision_micro'] = precision_micro
            metrics['recall_micro'] = recall_micro
            metrics['f1_micro'] = f1_micro
            
        else:
            # Single-label classification metrics
            from sklearn.metrics import classification_report
            
            # Basic accuracy
            metrics['accuracy'] = accuracy_score(labels, predictions)
            
            # Precision, Recall, F1 (macro and micro averages)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                labels, predictions, average='macro', zero_division=0
            )
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                labels, predictions, average='micro', zero_division=0
            )
            
            metrics['precision_macro'] = precision_macro
            metrics['recall_macro'] = recall_macro
            metrics['f1_macro'] = f1_macro
            metrics['precision_micro'] = precision_micro
            metrics['recall_micro'] = recall_micro
            metrics['f1_micro'] = f1_micro
            
            # Weighted averages for imbalanced classes
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
            
            metrics['precision_weighted'] = precision_weighted
            metrics['recall_weighted'] = recall_weighted
            metrics['f1_weighted'] = f1_weighted
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        # Initialize learning rate scheduler
        total_steps = len(train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Train F1 (macro): {train_metrics['f1_macro']:.4f}")
            print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            
            # Print subset accuracy only for multi-label classification
            if self.is_multilabel:
                print(f"Val Subset Accuracy: {val_metrics['subset_accuracy']:.4f}")
            else:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                model_path = os.path.join(self.models_output_dir, "best_model.pth")
                self.save_model(model_path)
                print(f"New best model saved! Val F1: {best_val_f1:.4f}")
                print(f"Model saved to: {model_path}")
    
    def save_model(self, filepath: str):
        """
        Save the trained model with full training state.
        
        Args:
            filepath: Path to save the model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.model.num_labels,
            'max_length': self.max_length,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
        
        # Save MLBinarizer classes if available
        if self.mlb is not None:
            save_dict['mlb_classes'] = self.mlb.classes_
        
        torch.save(save_dict, filepath)
        print(f"Model and training state saved to {filepath}")
    
    def load_model(self, filepath: str, num_labels: int):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            num_labels: Number of labels
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            self.model = BERTMultiLabelClassifier(
                checkpoint['model_name'], 
                checkpoint['num_labels']
            )
            self.model.to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load model, optimizer, and training state from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model architecture info
        num_labels = checkpoint['num_labels']
        self.model_name = checkpoint.get('model_name', self.model_name)
        
        # Initialize model with correct architecture
        dropout_rate = checkpoint.get('dropout_rate', self.dropout_rate)
        self.model = BERTMultiLabelClassifier(self.model_name, num_labels, dropout_rate=dropout_rate)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        
        # Load training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_metrics = checkpoint['train_metrics']
            self.val_metrics = checkpoint['val_metrics']
            print(f"Resumed training history from epoch {len(self.train_losses)}")
        
        # Load MLBinarizer if available
        if 'mlb_classes' in checkpoint:
            from sklearn.preprocessing import MultiLabelBinarizer
            self.mlb = MultiLabelBinarizer()
            self.mlb.classes_ = checkpoint['mlb_classes']
            print("Loaded label encoder from checkpoint")
        
        print(f"Successfully loaded checkpoint. Model has {num_labels} labels.")
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # F1 Score plot
        train_f1 = [m['f1_macro'] for m in self.train_metrics]
        val_f1 = [m['f1_macro'] for m in self.val_metrics]
        axes[0, 1].plot(epochs, train_f1, 'b-', label='Training F1')
        axes[0, 1].plot(epochs, val_f1, 'r-', label='Validation F1')
        axes[0, 1].set_title('F1 Score (Macro)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        
        # Accuracy plot (subset accuracy for multi-label, regular accuracy for single-label)
        if self.is_multilabel:
            train_acc = [m['subset_accuracy'] for m in self.train_metrics]
            val_acc = [m['subset_accuracy'] for m in self.val_metrics]
            accuracy_title = 'Subset Accuracy'
        else:
            train_acc = [m['accuracy'] for m in self.train_metrics]
            val_acc = [m['accuracy'] for m in self.val_metrics]
            accuracy_title = 'Accuracy'
            
        axes[1, 0].plot(epochs, train_acc, 'b-', label='Training Accuracy')
        axes[1, 0].plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        axes[1, 0].set_title(accuracy_title)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Fourth plot: Hamming Loss for multi-label, or additional metric for single-label
        if self.is_multilabel:
            train_hamming = [m['hamming_loss'] for m in self.train_metrics]
            val_hamming = [m['hamming_loss'] for m in self.val_metrics]
            axes[1, 1].plot(epochs, train_hamming, 'b-', label='Training Hamming Loss')
            axes[1, 1].plot(epochs, val_hamming, 'r-', label='Validation Hamming Loss')
            axes[1, 1].set_title('Hamming Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Hamming Loss')
            axes[1, 1].legend()
        else:
            # For single-label, show precision and recall
            train_precision = [m['precision_macro'] for m in self.train_metrics]
            val_precision = [m['precision_macro'] for m in self.val_metrics]
            train_recall = [m['recall_macro'] for m in self.train_metrics]
            val_recall = [m['recall_macro'] for m in self.val_metrics]
            
            axes[1, 1].plot(epochs, train_precision, 'b-', label='Training Precision')
            axes[1, 1].plot(epochs, val_precision, 'r-', label='Validation Precision')
            axes[1, 1].plot(epochs, train_recall, 'b--', label='Training Recall')
            axes[1, 1].plot(epochs, val_recall, 'r--', label='Validation Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        plots_dir = os.path.dirname(save_path)
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Training history plot saved to {save_path}")


def main():
    """Main function to run the training pipeline."""
    
    # Initialize trainer
    trainer = BERTTrainer(
        model_name="bert-base-uncased",
        max_length=512,
        batch_size=8,  # Reduced for memory efficiency
        learning_rate=2e-5,
        num_epochs=3,
        warmup_steps=500
    )
    
    # Load data
    train_df, test_df = trainer.load_data()
    
    # Prepare datasets
    train_loader, val_loader, test_loader = trainer.prepare_datasets(train_df, test_df)
    
    # Initialize model
    num_labels = len(trainer.mlb.classes_)
    trainer.initialize_model(num_labels)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\n=== FINAL TEST EVALUATION ===")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        print(f"Test {metric}: {value:.4f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
