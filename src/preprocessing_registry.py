from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import CountVectorizer

def bow_preprocessing(dataset, criteria_texts, hyper_config, vectorizer=None):
    
    if vectorizer is None:
        vectorizer = CountVectorizer()
        # Fit the vectorizer on dataset['text'] and criteria_texts
        all_texts = list(dataset['text']) + [text for texts in criteria_texts.values() for text in texts]
        vectorizer.fit(all_texts)

    if hyper_config['batch_size'] == 'N':
        batch_size = len(dataset)
    else:
        batch_size = hyper_config['batch_size']


    def vectorize(data_row):
        return {'embedding': torch.tensor(vectorizer.transform(data_row).todense(), dtype=torch.float32)}
    
    vectorized_dataset = dataset.map(lambda x: vectorize(x['text']),
                                    batched=True, batch_size=batch_size)
    
    vectorized_criteria = {problem_id: [torch.tensor(vectorizer.transform([crit_text]).todense(), dtype=torch.float32)
                                        for crit_text in criteria_texts[problem_id]]
                            for problem_id in criteria_texts.keys()}
    
    return vectorized_dataset, vectorized_criteria, vectorizer
    

def bert_preprocessing(dataset, criteria_texts, hyper_config, bert=None):
    
    if hyper_config['batch_size'] == 'N':
        batch_size = len(dataset)
    else:
        batch_size = hyper_config['batch_size']

    tokenizer = AutoTokenizer.from_pretrained(hyper_config['bert_model_name'])

    def tokenize(data_row):
        return tokenizer(data_row, padding='max_length', return_tensors='pt', truncation=True, max_length=512)


    tokenized_dataset = dataset.map(lambda x: tokenize(x['text']),
                                    batched=True, batch_size=batch_size)
    
    tokenized_criteria = {problem_id: [tokenizer(crit_text,
                                                 padding='max_length',
                                                 return_tensors='pt',
                                                 truncation=True,
                                                max_length=512)
                                        for crit_text in criteria_texts[problem_id]]
                            for problem_id in criteria_texts.keys()}
    
    if hyper_config.get('finetune', False) and bert is None:
        # if finetuning, just return tokens so we can embed in training loop
        # if a bert was provided, we want to embed with that
        return tokenized_dataset, tokenized_criteria
    
    if bert is None:
        bert = AutoModel.from_pretrained(hyper_config['bert_model_name'])
    bert.eval()

    process_with_bert = bert_applier(bert, gradients=False)
    embedded_dataset = tokenized_dataset.map(process_with_bert, batched=True, batch_size=batch_size)

    embedded_criteria = {problem_id: [bert(**crit_text).last_hidden_state[:, 0, :].squeeze() for crit_text in tokenized_criteria[problem_id]]
                            for problem_id in tokenized_criteria.keys()}

    return embedded_dataset, embedded_criteria


def bert_applier(bert, gradients=False):
    def process_with_bert(example):
        # Extract tokenized inputs
        inputs = {
            "input_ids": torch.tensor(example["input_ids"]), 
            "attention_mask": torch.tensor(example["attention_mask"]),
        }
        
        # Include token_type_ids if available
        if "token_type_ids" in example:
            inputs["token_type_ids"] = torch.tensor(example["token_type_ids"])

        # Run through BERT (disable gradient computation)
        if gradients:
            outputs = bert(**inputs)
        else:
            with torch.no_grad():
                outputs = bert(**inputs)

        # Extract last hidden state (CLS token embedding)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Shape: (hidden_dim)

        # Preserve other dataset features and add the BERT embedding
        example["embedding"] = cls_embedding
        return example
    
    return process_with_bert