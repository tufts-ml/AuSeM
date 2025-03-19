import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MultiDomainMultiCriteriaClassifier(nn.Module):
    def __init__(self, 
                 criteria_to_head_mapping: list,    # List of lists: Maps each criterion to one of the 8 heads
                 embedding_dim: int = 768,
                 num_heads: int = 8,
                 output_length: int = 16,
                 finetune: bool = True,
                 bert_model_name: str = None):
        super(MultiDomainMultiCriteriaClassifier, self).__init__()
        
        self.finetune = finetune

        if bert_model_name is not None and finetune:
            # even though non-finetune models use bert, we can do it all in preprocessing
            # Load BERT model
            self.bert = AutoModel.from_pretrained(bert_model_name)

            
            if not self.finetune:
                # Freeze all BERT layers:
                for param in self.bert.parameters():
                    param.requires_grad = False
        

        # Create 8 shared classification heads with sigmoid activation
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()  # Apply sigmoid after the linear layer for binary predictions
            ) for _ in range(num_heads)
        ])

        self.output_length = output_length

        # Map each criterion to a classification head
        self.criteria_to_head_mapping = criteria_to_head_mapping  # Shape: [problems][criteria_indices]

    def forward(self, dataset, criteria):


        if 'embedding' not in dataset.keys():
            dataset['embedding'] = self.bert(input_ids=dataset['input_ids'],
                                             attention_mask=dataset['attention_mask'],
                                             token_type_ids=dataset['token_type_ids']).last_hidden_state[:, 0, :].squeeze()

            criteria = {problem_id: [self.bert(**crit_text).last_hidden_state[:, 0, :].squeeze() for crit_text in criteria[problem_id]]
                                    for problem_id in criteria.keys()}
        
        batch_size = len(dataset['embedding'])
        outputs = []
        lengths = []


        for i in range(batch_size):
            problem_idx = dataset['problem_indices'][i]
            text_embs = dataset['embedding'][i] 

            criteria_indices = self.criteria_to_head_mapping[problem_idx]

            # Predict for each criterion using its mapped head
            problem_outputs = []

            for j, head_idx in enumerate(criteria_indices):
                combined_emb = text_embs + criteria[int(problem_idx)][j]  # Combine embeddings
                prediction = self.classification_heads[head_idx](combined_emb)  # Sigmoid applied inside head
                problem_outputs.append(prediction.squeeze())

            outputs.append(torch.stack(problem_outputs))
            lengths.append(len(problem_outputs))

        # Pad outputs to uniform length for efficient loss computation
        max_length = self.output_length
        padded_outputs = torch.zeros(batch_size, max_length, device=text_embs.device)
        mask = torch.zeros(batch_size, max_length, device=text_embs.device)

        for i, output in enumerate(outputs):
            length = lengths[i]
            if length > 0:
                padded_outputs[i, :length] = output
                mask[i, :length] = 1  # Mark valid predictions

        return padded_outputs, mask  # (batch_size, max_criteria), (batch_size, max_criteria)


def tokenize_inputs(tokenizer, texts, max_length=512):
    """Tokenizes a list of texts for efficient batching."""
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)


def compute_loss(predictions, targets, mask, criterion, params):
    """
    Computes masked loss, ensuring padded values do not affect training.

    Args:
        predictions (torch.Tensor): Model outputs (batch_size, max_criteria).
        targets (torch.Tensor): Ground truth labels (batch_size, max_criteria).
        mask (torch.Tensor): Binary mask indicating valid predictions.
        criterion (nn.Module): Loss function (e.g., nn.BCELoss(reduction='none')).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    loss, nll, bb_log_prob, clf_log_prob, unweighted_nll = criterion(predictions, targets, params)

    masked_loss = (loss * mask).sum() / mask.sum()  # Average over valid predictions
    final_loss = masked_loss + bb_log_prob + clf_log_prob

    return final_loss

