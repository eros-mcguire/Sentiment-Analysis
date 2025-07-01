# Sentiment Analysis with BERT on a Synthetic Dataset

Overview:
This project demonstrates a complete workflow for performing binary sentiment classification using a pre-trained BERT model. A small synthetic dataset of movie reviews was created to classify sentiments as positive or negative.

🔧 Technologies & Tools Used:
Language: Python

Libraries: PyTorch, Hugging Face Transformers, scikit-learn, pandas

Model: BERT (bert-base-uncased) for Sequence Classification

Hardware: GPU-accelerated training (if available)

📁 Project Workflow:
1. Data Preparation:
Created a synthetic dataset of 6 movie reviews with manually labeled sentiments (1: Positive, 0: Negative).

Stored and preprocessed data using pandas.

2. Tokenization & Encoding:
Utilized Hugging Face's BertTokenizer to tokenize text.

Generated input IDs and attention masks with encode_plus, padded to a maximum length of 256 tokens.

Converted encoded data into PyTorch tensors for model input.

3. Model Fine-Tuning:
Loaded the pre-trained BertForSequenceClassification with 2 output labels.

Fine-tuned the model for 2 epochs using AdamW optimizer.

Trained on mini-batches using PyTorch's DataLoader.

4. Evaluation:
Performed evaluation on the same dataset (demonstrative purpose).

Achieved an accuracy score of 100%, highlighting effective fine-tuning even with minimal data.

✅ Key Takeaways:
Learned how to fine-tune a pre-trained BERT model for text classification tasks.

Gained hands-on experience with tokenization, attention masks, and working with GPU acceleration in PyTorch.

Demonstrated end-to-end NLP pipeline from raw text to model evaluation.

📌 Potential Improvements:
Expand the dataset to improve generalization and prevent overfitting.

Introduce validation/testing splits for better performance assessment.

Apply early stopping and learning rate scheduling for optimized training.



