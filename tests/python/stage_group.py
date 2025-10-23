



def group_by_action( action, property, profit):
    """_summary_
    Args:
        property (1,N): _description_
        action (1,N): _description_
        profit (1,N): _description_
    """
    len= len(action)    
    table_act_property=map()
    table_profit=map()
    for i in range(len):
        table_act_property[action[i]] = table_act_property[action[i]] +(property[i], profit[i]*property[i]) # 加权平均
    return table_act_property, table_profit 

def cal_act_gain(prices,current_shares,remaining_cash):
    return d0_act, gain

def fn(action, property):
    profit =   cal_act_gain(action)
 
    table_loss= group_by_action(action, property, profit) # min (profit, 0) // 亏损期望
    table_gain=group_by_action(action, property, profit) #  E(profit) // 期望
    table_loss= group_by_action(action, property, profit) # max (profit, 0) // 收益期望
    
    # 排序，选择 亏损期望最小，平均期望较高， 收益期望较高的action

   