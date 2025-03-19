import yaml
import wandb
import torch
import numpy as np
import pandas as pd
from data_loader import load_datasets, create_hf_dataset
from preprocessing_registry import bert_preprocessing, bow_preprocessing, bert_applier
from classifier_model import MultiDomainMultiCriteriaClassifier, compute_loss
from loss_opt import initialize_loss, initialize_optimizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_experiment(hyper_config, problem_config):
    set_seed(hyper_config['seed'])

    # Initialize wandb if specified
    use_wandb = hyper_config.get('use_wandb', False)
    if use_wandb:
        wandb.init(project=hyper_config['wandb_project'], config=hyper_config, name=hyper_config['experiment_name'])

    # Load datasets
    train_datasets, val_datasets = load_datasets(hyper_config, problem_config, train=True, val=True, test=False)

    criteria_texts = train_datasets[0][-1]
    assert train_datasets[0][-1] == val_datasets[-1][-1], 'Criteria texts must be the same for train and val datasets'
    criteria_to_head_mapping = problem_config['criteria_to_head_mapping']
    
    # Convert each fold to a hf dataset
    train_datasets = [create_hf_dataset(ds)[0] for ds in train_datasets]
    val_datasets = [create_hf_dataset(ds)[0] for ds in val_datasets]

    # check if bert in name
    if 'bert' in hyper_config['embedder'].lower():
        preprocessor = bert_preprocessing
    elif hyper_config['embedder'] == 'bow':
        preprocessor = bow_preprocessing
    else:
        print(f'OOOPS! Embedder {hyper_config["embedder"]} not supported')

    

    for fold, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):

        batch_size = hyper_config['batch_size']
        num_epochs = hyper_config['num_epochs']
        if batch_size == 'N':
            batch_size = len(train_dataset)

        # Preprocess datasets
        with torch.no_grad():
            if hyper_config['embedder'] == 'bow':
                train_dataset, train_criteria, vectorizer = preprocessor(train_dataset, criteria_texts, hyper_config)
                val_dataset, val_criteria, _ = preprocessor(val_dataset, criteria_texts, hyper_config, vectorizer=vectorizer)
            else:
                train_dataset, train_criteria = preprocessor(train_dataset, criteria_texts, hyper_config)
                val_dataset, val_criteria = preprocessor(val_dataset, criteria_texts, hyper_config)
            
            train_loader = DataLoader(train_dataset.with_format("torch", device=hyper_config['device']), batch_size=batch_size, shuffle=True,)
            val_loader = DataLoader(val_dataset.with_format("torch", device=hyper_config['device']), batch_size=batch_size, shuffle=False)

        # send to device
        train_criteria = {k: [v.to(hyper_config['device']) for v in val] for k, val in train_criteria.items()}
        val_criteria = {k: [v.to(hyper_config['device']) for v in val] for k, val in val_criteria.items()}
        

        if 'embedding' in train_dataset.features:
            embedding_dim = len(train_dataset['embedding'][0])
        elif 'bert' in hyper_config['embedder'].lower():
            embedding_dim = 768
        else:
            raise ValueError('Embedding dimension not found')   

        model = MultiDomainMultiCriteriaClassifier(
            finetune=hyper_config.get('finetune', False),
            embedding_dim = embedding_dim,
            criteria_to_head_mapping=criteria_to_head_mapping,
            bert_model_name=hyper_config.get('bert_model_name', None),
            output_length=max([len(c_list) for c_list in criteria_to_head_mapping])
        ).to(hyper_config['device'])

        # Initialize loss and optimizer
        criterion = initialize_loss(hyper_config, model)
        optimizer = initialize_optimizer(hyper_config, model)
            
        train_metrics_list = []
        val_metrics_list = []
        for epoch in range(num_epochs):
            # time epoch
            import time
            start_time = time.time()
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                predictions, mask = model(batch, train_criteria)

                targets = batch['targets']
                params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
                
                loss = compute_loss(predictions, targets, mask, criterion, params)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            end_time = time.time()

            if (epoch) % hyper_config['val_freq'] == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    predictions_list, mask_list, targets_list = [], [], []
                    for batch in val_loader:
                        predictions, mask = model(batch, train_criteria)

                        targets = batch['targets']
                        params = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
                        
                        loss = compute_loss(predictions, targets, mask, criterion, params)
                        
                        val_loss += loss.item()
                        predictions_list.append(predictions)
                        mask_list.append(mask)
                        targets_list.append(targets)

                    predictions = torch.cat(predictions_list, dim=0)
                    targets = torch.cat(targets_list, dim=0)
                    mask = torch.cat(mask_list, dim=0)
                    numpy_mask = mask.cpu().numpy()
                    masked_predictions = predictions.cpu().numpy()[numpy_mask == 1]
                    masked_targets = targets.cpu().numpy()[numpy_mask == 1]
                    all_preds = masked_predictions.flatten()
                    all_labels = masked_targets.flatten()

                    threshold_preds = (np.array(all_preds) > 0.5).astype(int)

                    val_metrics = {
                        'accuracy': accuracy_score(all_labels, threshold_preds),
                        'precision': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[0],
                        'recall': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[1],
                        'auroc': roc_auc_score(all_labels, all_preds),
                        'auprc': average_precision_score(all_labels, all_preds)
                    }

                # Store metrics
                train_metrics_list.append({'epoch': epoch + 1, 'train_loss': train_loss})
                val_metrics_list.append({'epoch': epoch + 1, 'val_loss': val_loss, **val_metrics})

            print(f'Epoch {epoch} Train Loss: {train_loss}, time: {end_time - start_time}')

        torch.save(model.state_dict(), hyper_config['final_model_path'].format(fold=fold))

        train_df = pd.DataFrame(train_metrics_list)
        val_df = pd.DataFrame(val_metrics_list)
        
        train_csv_path = hyper_config['train_metrics_path'].format(fold=fold)
        val_csv_path = hyper_config['val_metrics_path'].format(fold=fold)

        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
                



    

    return



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with MultiDomainMultiCriteriaClassifier')
    parser.add_argument('--hyper_config', type=str, help='Path to hyperparameter configuration file')
    parser.add_argument('--problem_config', type=str, help='Path to problem configuration file')
    args = parser.parse_args()

    with open(args.hyper_config) as f:
        hyper_config = yaml.safe_load(f)

    with open(args.problem_config) as f:
        problem_config = yaml.safe_load(f)

    run_experiment(hyper_config, problem_config)
