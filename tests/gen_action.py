import torch 

import numpy as np

class myModule(torch.nn.Module):
    def __init__(self):
        super(myModule, self).__init__()
        self.linear = torch.nn.Linear(10, 5)


    """
    Args:
        pred_price_seqs: predict seq price, (nb_pred, seq_len)
        pred_property:  property (nb_pred)
        stock_holdings: current stock holdings
        remaining_cash: int, the Remaining investable funds
    Returns:
        action: torch.Tensor, the action to take
    """
    def forward(self, pred_price_seqs: torch.Tensor, pred_property: torch.Tensor,
                stock_holdings: int, remaining_cash: int):
        assert(remaining_cash >=0)
        assert(stock_holdings >=0)        
        # return self.linear(x)
        action = torch.zeros(1)  # buy, sell, hold
        return  action





def gen_action(price_seq, stock_holdings, remaining_cash):
    """
    Generate action based on model prediction and current state.
    Args:
        model: The trained model to generate actions.
        pred_price_seq: Predicted price sequence (seq_len)
        stock_holdings: Current stock holdings.
        remaining_cash: Remaining investable funds.
    Returns:
        action: The action to take (buy, sell, hold).
    """
    price_0= price_seq[0]
    
    
    