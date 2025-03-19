import torch


class L2SPLoss(torch.nn.Module):
    def __init__(self, alpha, bb_loc, beta, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.bb_loc = bb_loc
        self.beta = beta
        self.criterion = criterion
        self.D = len(self.bb_loc)

    def forward(self, logits, labels, params):

        unweighted_nll = self.criterion(logits, labels)
        nll = unweighted_nll

        bb_log_prob = (self.alpha/2) * ((params[:self.D] - self.bb_loc)**2).sum()
        clf_log_prob = (self.beta/2) * (params[self.D:]**2).sum()
        
        loss = nll #+ bb_log_prob + clf_log_prob
        return loss, nll, bb_log_prob, clf_log_prob, unweighted_nll.mean()
    
class WeightedBCELoss(torch.nn.Module,):
    def __init__(self, criterion=torch.nn.BCELoss(reduction='none')):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, weights, params):
        unweighted_nll = self.criterion(logits, labels)
        nll = unweighted_nll*weights
        nll = nll.mean()
        unweighted_nll = unweighted_nll.mean()
        # return 0 tensors for the other loss components
        return nll, torch.tensor(0), torch.tensor(0), torch.tensor(0), unweighted_nll
    
class UnWeightedBCELoss(torch.nn.Module,):
    def __init__(self, criterion=torch.nn.BCELoss(reduction='none')):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, params):
        nll = self.criterion(logits, labels)
        nll = nll.mean()
        unweighted_nll = nll.mean()
        # return 0 tensors for the other loss components
        return nll, torch.tensor(0), torch.tensor(0), torch.tensor(0), unweighted_nll

def initialize_loss(hyper_config, model):

    if hyper_config['loss'] == 'l2sp':
        params = torch.nn.utils.parameters_to_vector(model.bert.parameters()).detach()
        bb_loc = params
        return L2SPLoss(hyper_config['alpha'], bb_loc, hyper_config['beta'], criterion=torch.nn.BCELoss(reduction='none'))
    else:
        return UnWeightedBCELoss()
    

def initialize_optimizer(hyper_config, model):
    """
    Initializes the optimizer. If fine-tuning is enabled, includes all model parameters.
    If fine-tuning is disabled, excludes BERT parameters and optimizes only classification heads.

    Args:
        hyper_config (dict): Hyperparameter configuration.
        model (MultiDomainMultiCriteriaClassifier): The model to optimize.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    learning_rate = hyper_config['learning_rate']
    weight_decay = hyper_config['opt_weight_decay']

    if hyper_config.get('finetune', False):
        # Fine-tune all model parameters, including BERT
        params = model.parameters()
    else:
        # Freeze BERT and only optimize classification heads
        params = model.classification_heads.parameters()

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    return optimizer
