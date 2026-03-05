import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_CONFIG, TRAINING_CONFIG, TOPICS_CONFIG


class ForumDataProcessor:
    """Processes forum data for multi-label classification."""
    
    def __init__(self, dataset_path: str = None, topics_file: str = None, target_field: str = None):
        """
        Initialize the data processor.
        
        Args:
            dataset_path: Path to the dataset directory containing JSON files (defaults to config)
            topics_file: Path to the topics mapping JSON file (defaults to config)
            target_field: Target field for classification: "topics", "branch_status", "branch_type", or "overall_thread_sentiment" (defaults to config)
        """
        self.dataset_path = dataset_path or DATA_CONFIG["dataset_path"]
        self.topics_file = topics_file or TOPICS_CONFIG["topics_file"]
        self.target_field = target_field or DATA_CONFIG["target_field"]
        self.data = []
        self.df = None
        self.mlb = MultiLabelBinarizer()
        self.topics_mapping = self._load_topics_mapping()
        
    def _load_topics_mapping(self) -> Dict[str, str]:
        """
        Load the topics mapping from topics.json file.
        
        Returns:
            Dictionary mapping topic IDs to topic names
        """
        try:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
                topics_mapping = topics_data.get('topics', {})
                print(f"Loaded {len(topics_mapping)} topic categories from {self.topics_file}")
                return topics_mapping
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load topics file {self.topics_file}: {e}")
            print("Using topic IDs as labels instead of names")
            return {}
    
    def get_topic_name(self, topic_id: str) -> str:
        """
        Get topic name from topic ID.
        
        Args:
            topic_id: Topic ID string
            
        Returns:
            Topic name or ID if mapping not found
        """
        return self.topics_mapping.get(topic_id, topic_id)

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load all JSON files from the dataset directory.
        
        Returns:
            List of processed forum data entries
        """
        print(f"Loading data from {self.dataset_path}...")
        json_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files")
        
        for i, filename in enumerate(json_files):
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(json_files)}")
                
            file_path = os.path.join(self.dataset_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed_entry = self._process_entry(data)
                    if processed_entry:
                        self.data.append(processed_entry)
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"Error processing {filename}: {e}")
                continue
                
        print(f"Successfully loaded {len(self.data)} entries")
        return self.data
    
    def _process_entry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single forum entry.
        
        Args:
            data: Raw forum data from JSON file
            
        Returns:
            Processed entry with merged comment text and topic labels
        """
        # Extract metadata
        metadata = data.get('metadata', {})
        content = data.get('content', [])
        topics = data.get('topics', {})
        analysis = data.get('analysis', {})
        
        # Merge all comment bodies into a single text
        comment_texts = []
        for comment in content:
            comment_body = comment.get('comment_body', '')
            if comment_body:
                # Clean and preprocess text
                cleaned_text = self._clean_text(comment_body)
                comment_texts.append(cleaned_text)
        
        if not comment_texts:
            return None
            
        merged_text = ' '.join(comment_texts)
        
        # Collect ALL possible target labels for universal CSV
        topics_labels = []
        if topics:
            for topic_id in topics.keys():
                topics_labels.append(topic_id)
        
        branch_status_labels = []
        branch_status = metadata.get('branch_status')
        if branch_status:
            branch_status_labels.append(branch_status)
        
        branch_type_labels = []
        branch_type = metadata.get('branch_type')
        if branch_type:
            branch_type_labels.append(branch_type)
        
        sentiment_labels = []
        sentiment = analysis.get('overall_thread_sentiment')
        if sentiment:
            sentiment_labels.append(sentiment)
        
        # For backward compatibility, set topic_labels to the current target
        if self.target_field == "topics":
            current_labels = topics_labels
        elif self.target_field == "branch_status":
            current_labels = branch_status_labels
        elif self.target_field == "branch_type":
            current_labels = branch_type_labels
        elif self.target_field == "overall_thread_sentiment":
            current_labels = sentiment_labels
        else:
            raise ValueError(f"Invalid target_field: {self.target_field}. Must be 'topics', 'branch_status', 'branch_type', or 'overall_thread_sentiment'")
        
        return {
            'branch_id': metadata.get('branch_id'),
            'thread_id': metadata.get('thread_id'),
            'publication_year': metadata.get('publication_year'),
            'text': merged_text,
            'topics_labels': topics_labels,  # Topic IDs
            'branch_status_labels': branch_status_labels,  # Status labels
            'branch_type_labels': branch_type_labels,  # Type labels
            'sentiment_labels': sentiment_labels,  # Sentiment labels
            'num_comments': len(content),
            'text_length': len(merged_text),
            'insight_summary': analysis.get('insight_summary', '')
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def filter_data(self) -> pd.DataFrame:
        """
        Filter data based on configuration parameters.
        
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("DataFrame not created. Please run create_dataframe() first.")
        
        original_count = len(self.df)
        filtered_df = self.df.copy()
        
        # Apply text length filtering if configured
        min_length = DATA_CONFIG.get("min_text_length", 0)
        max_length = DATA_CONFIG.get("max_text_length", float('inf'))
        
        if min_length > 0:
            filtered_df = filtered_df[filtered_df['text_length'] >= min_length]
            print(f"Filtered {original_count - len(filtered_df)} samples with text length < {min_length}")
        
        if max_length < float('inf'):
            filtered_df = filtered_df[filtered_df['text_length'] <= max_length]
            print(f"Filtered {len(self.df) - len(filtered_df)} samples with text length > {max_length}")
        
        # Apply minimum labels filtering if configured
        min_labels = DATA_CONFIG.get("min_labels_per_sample", 1)
        if min_labels > 0:
            # Use the current target's labels column for filtering
            if self.target_field == "topics":
                labels_column = 'topics_labels'
            elif self.target_field == "branch_status":
                labels_column = 'branch_status_labels'
            elif self.target_field == "branch_type":
                labels_column = 'branch_type_labels'
            elif self.target_field == "overall_thread_sentiment":
                labels_column = 'sentiment_labels'
            else:
                labels_column = 'topics_labels'  # Default fallback
            
            if labels_column in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[labels_column].apply(len) >= min_labels]
                print(f"Filtered {len(self.df) - len(filtered_df)} samples with < {min_labels} labels")
        
        print(f"Data filtering completed: {original_count} -> {len(filtered_df)} samples")
        self.df = filtered_df
        return self.df
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Convert loaded data to pandas DataFrame.
        
        Returns:
            DataFrame with processed forum data
        """
        if not self.data:
            raise ValueError("No data loaded. Please run load_data() first.")
            
        self.df = pd.DataFrame(self.data)
        return self.df
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        if self.df is None:
            raise ValueError("DataFrame not created. Please run create_dataframe() first.")
            
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(self.df)
        analysis['avg_text_length'] = self.df['text_length'].mean()
        analysis['median_text_length'] = self.df['text_length'].median()
        analysis['avg_comments_per_thread'] = self.df['num_comments'].mean()
        
        # Label analysis based on current target field
        if self.target_field == "topics":
            labels_column = 'topics_labels'
        elif self.target_field == "branch_status":
            labels_column = 'branch_status_labels'
        elif self.target_field == "branch_type":
            labels_column = 'branch_type_labels'
        elif self.target_field == "overall_thread_sentiment":
            labels_column = 'sentiment_labels'
        else:
            labels_column = 'topics_labels'  # Default fallback
        
        all_labels = []
        if labels_column in self.df.columns:
            for labels in self.df[labels_column]:
                all_labels.extend(labels)
        
        label_counts = Counter(all_labels)
        analysis['unique_labels'] = len(label_counts)
        analysis['most_common_labels'] = label_counts.most_common(10)
        analysis['label_distribution'] = dict(label_counts)
        analysis['target_field'] = getattr(self, 'target_field', 'topics')  # Include target field info
        
        # Label statistics per sample
        labels_per_sample = []
        if labels_column in self.df.columns:
            labels_per_sample = [len(labels) for labels in self.df[labels_column]]
        
        analysis['avg_labels_per_sample'] = np.mean(labels_per_sample) if labels_per_sample else 0
        analysis['max_labels_per_sample'] = np.max(labels_per_sample) if labels_per_sample else 0
        analysis['min_labels_per_sample'] = np.min(labels_per_sample) if labels_per_sample else 0
        
        # Year distribution
        year_counts = self.df['publication_year'].value_counts().to_dict()
        analysis['year_distribution'] = year_counts
        
        # Status and type distribution from label columns
        if 'branch_status_labels' in self.df.columns:
            status_labels = []
            for labels in self.df['branch_status_labels']:
                status_labels.extend(labels)
            status_counts = Counter(status_labels)
            analysis['status_distribution'] = dict(status_counts)
        
        if 'branch_type_labels' in self.df.columns:
            type_labels = []
            for labels in self.df['branch_type_labels']:
                type_labels.extend(labels)
            type_counts = Counter(type_labels)
            analysis['type_distribution'] = dict(type_counts)
        
        return analysis
    
    def prepare_for_training(self, test_size: float = None, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training by encoding labels and splitting.
        
        Args:
            test_size: Fraction of data to use for testing (defaults to config)
            random_state: Random seed for reproducibility (defaults to config)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if test_size is None:
            test_size = TRAINING_CONFIG["test_size"]
        if random_state is None:
            random_state = TRAINING_CONFIG["random_state"]
            
        if self.df is None:
            raise ValueError("DataFrame not created. Please run create_dataframe() first.")
        
        # Select the appropriate labels column based on target field
        if self.target_field == "topics":
            labels_column = 'topics_labels'
        elif self.target_field == "branch_status":
            labels_column = 'branch_status_labels'
        elif self.target_field == "branch_type":
            labels_column = 'branch_type_labels'
        elif self.target_field == "overall_thread_sentiment":
            labels_column = 'sentiment_labels'
        else:
            # Use the specific labels column directly
            labels_column = 'topics_labels'  # Default fallback
        
        # Check if the column exists
        if labels_column not in self.df.columns:
            raise ValueError(f"Labels column '{labels_column}' not found. Available columns: {list(self.df.columns)}")
        
        # Filter out samples with no labels
        df_with_labels = self.df[self.df[labels_column].apply(len) > 0].copy()
        print(f"Samples with labels: {len(df_with_labels)}/{len(self.df)}")
        print(f"Target field used: {self.target_field} (using column: {labels_column})")
        
        # Collect all unique labels that appear in the data
        all_labels = set()
        for labels in df_with_labels[labels_column]:
            all_labels.update(labels)
        
        # Handle different target fields
        if self.target_field == "topics":
            # Multi-label classification for topics - sort numerically
            ordered_classes = sorted([label for label in all_labels], key=int)
        else:
            # Single-label classification for branch_status, branch_type, or overall_thread_sentiment - sort alphabetically
            ordered_classes = sorted(list(all_labels))
        
        print(f"Found {len(ordered_classes)} unique labels: {ordered_classes}")
        
        # For single-label classification (branch_status, branch_type, overall_thread_sentiment), 
        # we still use MultiLabelBinarizer for consistency, but each sample will have only one label
        self.mlb.classes_ = np.array(ordered_classes)
        
        # Create a mapping from class to index for transformation
        class_to_index = {cls: i for i, cls in enumerate(ordered_classes)}
        
        # Manual transformation to maintain our class order
        n_samples = len(df_with_labels)
        n_classes = len(ordered_classes)
        encoded_labels = np.zeros((n_samples, n_classes), dtype=int)
        
        for i, labels in enumerate(df_with_labels[labels_column]):
            for label in labels:
                if label in class_to_index:
                    encoded_labels[i, class_to_index[label]] = 1
        
        # For single-label classification, verify each sample has exactly one label
        if self.target_field in ["branch_status", "branch_type", "overall_thread_sentiment"]:
            labels_per_sample = encoded_labels.sum(axis=1)
            if np.any(labels_per_sample != 1):
                print(f"Warning: Some samples have {np.unique(labels_per_sample)} labels. Expected exactly 1 for {self.target_field}.")
        
        # Split the data with fixed random state for reproducibility
        train_df, test_df = train_test_split(
            df_with_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # Can't stratify multi-label directly
        )
        
        # Add split information for tracking
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Number of unique labels: {len(self.mlb.classes_)}")
        print(f"Target field: {self.target_field}")
        print(f"Random state used: {random_state}")
        
        return train_df, test_df
    
    def calculate_class_weights(self, train_df: pd.DataFrame, method: str = "balanced") -> np.ndarray:
        """
        Calculate class weights for imbalanced multi-label classification.
        
        Args:
            train_df: Training dataframe with universal label columns
            method: Method for calculating weights ('balanced', 'inverse_freq', 'custom')
            
        Returns:
            Array of class weights for each label
        """
        if self.mlb is None:
            raise ValueError("MultiLabelBinarizer not initialized. Run prepare_for_training first.")
        
        # Select the appropriate labels column based on target field
        if self.target_field == "topics":
            labels_column = 'topics_labels'
        elif self.target_field == "branch_status":
            labels_column = 'branch_status_labels'
        elif self.target_field == "branch_type":
            labels_column = 'branch_type_labels'
        elif self.target_field == "overall_thread_sentiment":
            labels_column = 'sentiment_labels'
        else:
            labels_column = 'topics_labels'  # Default fallback
        
        # Calculate positive counts for each label by counting occurrences in the labels column
        label_counts = []
        total_samples = len(train_df)
        
        for label in self.mlb.classes_:
            count = 0
            for sample_labels in train_df[labels_column]:
                if label in sample_labels:
                    count += 1
            label_counts.append(count)
        
        label_counts = np.array(label_counts)
        
        if method == "balanced":
            # Balanced weights: total_samples / (2 * positive_count)
            # This gives higher weight to less frequent classes
            weights = np.where(label_counts > 0, 
                             total_samples / (2.0 * label_counts), 
                             1.0)
        elif method == "inverse_freq":
            # Inverse frequency: 1 / frequency
            frequencies = label_counts / total_samples
            weights = np.where(frequencies > 0, 1.0 / frequencies, 1.0)
        elif method == "custom":
            # Custom weights: sqrt(total_samples / positive_count)
            # Less aggressive than balanced but still helps
            weights = np.where(label_counts > 0,
                             np.sqrt(total_samples / label_counts),
                             1.0)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Normalize weights to have mean of 1.0
        weights = weights / np.mean(weights)
        
        print(f"Class weights calculated using method '{method}':")
        for i, (label, weight, count) in enumerate(zip(self.mlb.classes_, weights, label_counts)):
            print(f"  Label {label}: weight={weight:.3f}, samples={count}/{total_samples} ({count/total_samples*100:.1f}%)")
        
        return weights
    
    def visualize_data(self, save_path: str = "data_analysis.png"):
        """
        Create visualizations of the data distribution.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.df is None:
            raise ValueError("DataFrame not created. Please run create_dataframe() first.")
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forum Data Analysis', fontsize=16)
        
        # Text length distribution
        axes[0, 0].hist(self.df['text_length'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Number of comments per thread
        axes[0, 1].hist(self.df['num_comments'], bins=30, edgecolor='black')
        axes[0, 1].set_title('Comments per Thread Distribution')
        axes[0, 1].set_xlabel('Number of Comments')
        axes[0, 1].set_ylabel('Frequency')
        
        # Labels per sample
        # Use the current target's labels column
        if self.target_field == "topics":
            labels_column = 'topics_labels'
        elif self.target_field == "branch_status":
            labels_column = 'branch_status_labels'
        elif self.target_field == "branch_type":
            labels_column = 'branch_type_labels'
        elif self.target_field == "overall_thread_sentiment":
            labels_column = 'sentiment_labels'
        else:
            labels_column = 'topics_labels'  # Default fallback
        
        labels_per_sample = []
        if labels_column in self.df.columns:
            labels_per_sample = [len(topics) for topics in self.df[labels_column]]
        
        if labels_per_sample:
            axes[0, 2].hist(labels_per_sample, bins=20, edgecolor='black')
        axes[0, 2].set_title('Labels per Sample Distribution')
        axes[0, 2].set_xlabel('Number of Labels')
        axes[0, 2].set_ylabel('Frequency')
        
        # Year distribution
        year_counts = self.df['publication_year'].value_counts().sort_index()
        axes[1, 0].bar(year_counts.index, year_counts.values)
        axes[1, 0].set_title('Publication Year Distribution')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Number of Threads')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Topic status distribution
        if 'branch_status_labels' in self.df.columns:
            status_labels = []
            for labels in self.df['branch_status_labels']:
                status_labels.extend(labels)
            if status_labels:
                status_counts = Counter(status_labels)
                axes[1, 1].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Branch Status Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Status Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Top 10 most common topics
        all_topics = []
        if labels_column in self.df.columns:
            for topics in self.df[labels_column]:
                all_topics.extend(topics)
        
        if all_topics:
            topic_counts = Counter(all_topics)
            top_topics = dict(topic_counts.most_common(10))
            
            axes[1, 2].barh(list(top_topics.keys()), list(top_topics.values()))
        axes[1, 2].set_title(f'Top 10 Most Common {self.target_field.replace("_", " ").title()}')
        axes[1, 2].set_xlabel('Frequency')
        
        plt.tight_layout()
        
        # Ensure plots directory exists (only if save_path contains a directory)
        dir_path = os.path.dirname(save_path)
        if dir_path:  # Only create directory if path is not empty
            os.makedirs(dir_path, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Data visualization saved to {save_path}")
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           output_dir: str = None):
        """
        Save processed data to files. Train/test CSVs are shared across all targets,
        only label encoder and class info are target-specific.
        For topics, also add one-hot encoded columns (topic_id_1, topic_id_2, etc.).
        
        Args:
            train_df: Training data DataFrame
            test_df: Test data DataFrame  
            output_dir: Directory to save processed data (defaults to config)
        """
        if output_dir is None:
            output_dir = DATA_CONFIG["processed_data_dir"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create copies of the dataframes to avoid modifying the originals
        train_df_to_save = train_df.copy()
        test_df_to_save = test_df.copy()
        
        # Add one-hot encoded columns for topics
        if self.target_field == "topics" and self.mlb is not None:
            print("Adding one-hot encoded topic columns...")
            
            # Get topic labels for train and test
            train_topic_labels = train_df_to_save['topics_labels'].tolist()
            test_topic_labels = test_df_to_save['topics_labels'].tolist()
            
            # Create one-hot encoding manually to avoid sklearn version issues
            # Get all unique topic classes from the labels
            all_topics = set()
            for label_list in train_topic_labels + test_topic_labels:
                all_topics.update(label_list)
            
            # Sort topics numerically (convert to int for proper numeric sorting)
            try:
                topic_classes = sorted(list(all_topics), key=lambda x: int(x))
            except ValueError:
                # Fallback to string sorting if not all are numeric
                topic_classes = sorted(list(all_topics))
            
            print(f"Found {len(topic_classes)} unique topic classes: {topic_classes[:10]}{'...' if len(topic_classes) > 10 else ''}")
            
            # Create one-hot columns for each topic class
            for topic_class in topic_classes:
                col_name = f"topic_id_{topic_class}"
                
                # For train data - use int values (1, 0)
                train_df_to_save[col_name] = [
                    1 if topic_class in label_list else 0 
                    for label_list in train_topic_labels
                ]
                
                # For test data - use int values (1, 0)
                test_df_to_save[col_name] = [
                    1 if topic_class in label_list else 0 
                    for label_list in test_topic_labels
                ]
            
            print(f"Added {len(topic_classes)} one-hot encoded topic columns")
        
        # Save dataframes (shared across all target fields)
        train_df_to_save.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
        test_df_to_save.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
        
        # Create file names with target field suffix for label-specific files
        target_suffix = f"_{self.target_field}"
        
        # Save label encoder (target-specific)
        import pickle
        with open(os.path.join(output_dir, f"label_encoder{target_suffix}.pkl"), "wb") as f:
            pickle.dump(self.mlb, f)
            
        # Save class information with target field details (target-specific)
        class_info = {
            'classes': list(self.mlb.classes_),
            'num_classes': len(self.mlb.classes_),
            'target_field': self.target_field,
            'description': f'Labels for {self.target_field} classification'
        }
        with open(os.path.join(output_dir, f"class_info{target_suffix}.json"), "w") as f:
            json.dump(class_info, f, indent=2)
            
        print(f"Processed data for target '{self.target_field}' saved to {output_dir}/")
        print(f"Shared files: train_data.csv, test_data.csv")
        print(f"Target-specific files: label_encoder{target_suffix}.pkl, class_info{target_suffix}.json")
        
        if self.target_field == "topics":
            topic_columns = [f"topic_id_{topic_class}" for topic_class in self.mlb.classes_]
            print(f"One-hot topic columns: {topic_columns[:5]}{'...' if len(topic_columns) > 5 else ''}")
    
    def load_processed_data(self, data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved processed data splits. Train/test CSVs are shared,
        only label encoder is target-specific.
        
        Args:
            data_dir: Directory containing processed data (defaults to config)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if data_dir is None:
            data_dir = DATA_CONFIG["processed_data_dir"]
            
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
        
        # Load shared train/test files
        train_file = os.path.join(data_dir, "train_data.csv")
        test_file = os.path.join(data_dir, "test_data.csv")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Train or test data files not found in {data_dir}")
        
        # Load target-specific label encoder
        target_suffix = f"_{self.target_field}"
        encoder_file = os.path.join(data_dir, f"label_encoder{target_suffix}.pkl")
        
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Label encoder file not found for target '{self.target_field}' in {data_dir}")
        
        # Load the data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Load the target-specific label encoder
        import pickle
        with open(encoder_file, "rb") as f:
            self.mlb = pickle.load(f)
        
        print(f"Loaded processed data for target '{self.target_field}' from {data_dir}/")
        print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}, Classes: {len(self.mlb.classes_)}")
        
        # Convert all label columns back to lists (they were saved as strings)
        import ast
        label_columns_to_convert = ['topics_labels', 'branch_status_labels', 'branch_type_labels', 'sentiment_labels']
        
        for col in label_columns_to_convert:
            if col in train_df.columns:
                train_df[col] = train_df[col].apply(ast.literal_eval)
                test_df[col] = test_df[col].apply(ast.literal_eval)
        
        return train_df, test_df
    
    def build_encoder_for_target(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Build the MultiLabelBinarizer for the current target field from universal CSV data.
        This allows switching targets without regenerating the train/test split.
        
        Args:
            train_df: Training DataFrame with universal label columns
            test_df: Test DataFrame with universal label columns
        """
        # Select the appropriate labels column based on target field
        if self.target_field == "topics":
            labels_column = 'topics_labels'
        elif self.target_field == "branch_status":
            labels_column = 'branch_status_labels'
        elif self.target_field == "branch_type":
            labels_column = 'branch_type_labels'
        elif self.target_field == "overall_thread_sentiment":
            labels_column = 'sentiment_labels'
        else:
            # Default to topics
            labels_column = 'topics_labels'
        
        # Check if the column exists
        if labels_column not in train_df.columns:
            raise ValueError(f"Labels column '{labels_column}' not found. Available columns: {list(train_df.columns)}")
        
        # Combine train and test data to get all possible labels
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Filter out samples with no labels for this target
        df_with_labels = combined_df[combined_df[labels_column].apply(len) > 0]
        
        # Collect all unique labels that appear in the data
        all_labels = set()
        for labels in df_with_labels[labels_column]:
            all_labels.update(labels)
        
        # Handle different target fields
        if self.target_field == "topics":
            # Multi-label classification for topics - sort numerically
            ordered_classes = sorted([label for label in all_labels], key=int)
        else:
            # Single-label classification for branch_status, branch_type, or overall_thread_sentiment - sort alphabetically
            ordered_classes = sorted(list(all_labels))
        
        print(f"Built encoder for target '{self.target_field}'" with {len(ordered_classes)} classes: {ordered_classes}")
        
        # Initialize the MultiLabelBinarizer with the ordered classes
        self.mlb.classes_ = np.array(ordered_classes)
        
        # Save the encoder for this target field
        output_dir = DATA_CONFIG["processed_data_dir"]
        target_suffix = f"_{self.target_field}"
        
        import pickle
        encoder_file = os.path.join(output_dir, f"label_encoder{target_suffix}.pkl")
        with open(encoder_file, "wb") as f:
            pickle.dump(self.mlb, f)
        
        # Save class information
        class_info = {
            'classes': list(self.mlb.classes_),
            'num_classes': len(self.mlb.classes_),
            'target_field': self.target_field,
            'description': f'Labels for {self.target_field} classification'
        }
        class_info_file = os.path.join(output_dir, f"class_info{target_suffix}.json")
        with open(class_info_file, "w") as f:
            json.dump(class_info, f, indent=2)
        
        print(f"Saved encoder and class info for target '{self.target_field}' to {output_dir}/")
    
    def get_or_create_training_data(self, data_split: bool = None, 
                                   data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get existing training data or create new split based on configuration.
        
        Args:
            data_split: Whether to create new split (True) or use existing only (False). 
                       If None, uses config default.
            data_dir: Directory containing processed data (defaults to config)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if data_split is None:
            data_split = DATA_CONFIG["data_split"]
        if data_dir is None:
            data_dir = DATA_CONFIG["processed_data_dir"]
            
        existing_data_available = os.path.exists(data_dir)
        
        # If data_split is False, only use existing data
        if not data_split:
            if existing_data_available:
                try:
                    print(f"Loading existing processed data (data_split={data_split})...")
                    return self.load_processed_data(data_dir)
                except FileNotFoundError as e:
                    # Check if we have universal CSV files but missing target-specific encoder
                    train_file = os.path.join(data_dir, "train_data.csv")
                    test_file = os.path.join(data_dir, "test_data.csv")
                    
                    if os.path.exists(train_file) and os.path.exists(test_file):
                        print(f"Found universal CSV files but missing encoder for target '{self.target_field}'")
                        print("Building encoder from existing data...")
                        
                        # Load the CSV files
                        train_df = pd.read_csv(train_file)
                        test_df = pd.read_csv(test_file)
                        
                        # Convert label columns back to lists
                        import ast
                        for col in ['topics_labels', 'branch_status_labels', 'branch_type_labels', 'sentiment_labels']:
                            if col in train_df.columns:
                                train_df[col] = train_df[col].apply(ast.literal_eval)
                                test_df[col] = test_df[col].apply(ast.literal_eval)
                        
                        # Build encoder for current target
                        self.build_encoder_for_target(train_df, test_df)
                        
                        return train_df, test_df
                    else:
                        print(f"Could not load existing split data: {e}")
                        print("data_split=False: No existing train/test split found, returning None")
                        return None, None
                except Exception as e:
                    print(f"Could not load existing split data: {e}")
                    print("data_split=False: No existing train/test split found, returning None")
                    return None, None
            else:
                print(f"data_split=False: No processed data directory found, returning None")
                return None, None
        
        # If data_split is True, always create new split
        print(f"Creating new data split (data_split={data_split})...")
        
        # Create new split - need to load and process raw data first
        if self.data is None or len(self.data) == 0:
            self.load_data()
        if self.df is None:
            self.create_dataframe()
            
        train_df, test_df = self.prepare_for_training()
        
        # Save the processed data for future use
        self.save_processed_data(train_df, test_df, data_dir)
        
        return train_df, test_df

def main():
    """Main function to run data processing pipeline."""
    # Initialize processor
    processor = ForumDataProcessor()
    
    # Get or create training data based on config first
    train_df, test_df = processor.get_or_create_training_data()
    
    # If we don't have split data but data_split=False, we can still do analysis and create metadata
    if train_df is None and test_df is None:
        print("No train/test split available or created (data_split=False)")
        print("Proceeding with data analysis and metadata creation...")
        
        # Load and process data for analysis
        data = processor.load_data()
        df = processor.create_dataframe()
        
        # Apply data filtering based on configuration
        processor.filter_data()
        
        # Analyze data
        analysis = processor.analyze_data()
        print("\n=== DATA ANALYSIS ===")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        # Create visualizations
        processor.visualize_data()
        
        print(f"\nData analysis completed successfully!")
        print(f"Raw data samples: {len(processor.df) if processor.df is not None else 'N/A'}")
        print(f"Target field: {processor.target_field}")
        print(f"Data split setting: {DATA_CONFIG['data_split']} (no train/test split created)")
        
    else:
        # We have training data, check if we need additional analysis
        analysis_needed = (DATA_CONFIG["data_split"] or 
                          not os.path.exists(os.path.join(DATA_CONFIG["processed_data_dir"], "data_analysis.json")))
        
        if analysis_needed:
            # Ensure we have the raw data loaded for analysis
            if processor.data is None or len(processor.data) == 0:
                data = processor.load_data()
            if processor.df is None:
                df = processor.create_dataframe()
            
            # Apply data filtering based on configuration
            processor.filter_data()
            
            # Analyze data
            analysis = processor.analyze_data()
            print("\n=== DATA ANALYSIS ===")
            for key, value in analysis.items():
                print(f"{key}: {value}")
            
            # Create visualizations
            processor.visualize_data()
        else:
            print("Using existing processed data and skipping analysis (data_split=False)")
        
        print(f"\nData processing completed successfully!")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Target field: {processor.target_field}")
        print(f"Data split setting: {DATA_CONFIG['data_split']}")


if __name__ == "__main__":
    main()
