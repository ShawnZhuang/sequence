import akshare as ak
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import logging

class SmartDataFetcher:
    def __init__(self, data_dir: str = "local_data"):
        """
        初始化智能数据获取器
        
        Args:
            data_dir: 本地数据存储目录
        """
        self.data_dir = data_dir
        self.setup_logging()
        self.ensure_data_dir()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.info(f"创建数据目录: {self.data_dir}")
    
    def get_local_filepath(self, data_type: str, symbol: str, **kwargs) -> str:
        """
        生成本地文件路径
        
        Args:
            data_type: 数据类型 (stock_daily, stock_info, etc.)
            symbol: 股票代码或其他标识符
            **kwargs: 其他参数用于生成文件名
            
        Returns:
            本地文件路径
        """
        # 清理symbol中的特殊字符
        clean_symbol = symbol.replace("/", "_").replace("\\", "_")
        
        if data_type == "stock_daily":
            # 对于股票日线数据，可以包含开始和结束日期
            start_date = kwargs.get('start_date', '')
            end_date = kwargs.get('end_date', '')
            if start_date and end_date:
                filename = f"{clean_symbol}_{start_date}_{end_date}.csv"
            else:
                filename = f"{clean_symbol}_all.csv"
        else:
            filename = f"{data_type}_{clean_symbol}.csv"
            
        return os.path.join(self.data_dir, filename)
    
    def is_data_fresh(self, filepath: str, max_age_hours: int = 24) -> bool:
        """
        检查数据是否新鲜（在指定时间内）
        
        Args:
            filepath: 文件路径
            max_age_hours: 最大允许的小时数
            
        Returns:
            数据是否新鲜
        """
        if not os.path.exists(filepath):
            return False
            
        file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        current_time = datetime.now()
        age_hours = (current_time - file_mtime).total_seconds() / 3600
        
        return age_hours <= max_age_hours
    
    def load_local_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        从本地加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            DataFrame 或 None（如果文件不存在）
        """
        try:
            if os.path.exists(filepath):
                self.logger.info(f"从本地加载数据: {filepath}")
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            return None
        except Exception as e:
            self.logger.error(f"加载本地数据失败 {filepath}: {e}")
            return None
    
    def save_local_data(self, data: pd.DataFrame, filepath: str):
        """
        保存数据到本地
        
        Args:
            data: 要保存的DataFrame
            filepath: 文件路径
        """
        try:
            data.to_csv(filepath)
            self.logger.info(f"数据已保存到本地: {filepath}")
        except Exception as e:
            self.logger.error(f"保存数据失败 {filepath}: {e}")
    
    def fetch_stock_daily(self, symbol: str, start_date: str = "20200101", 
                         end_date: str = "20231231", 
                         adjust: str = "", 
                         force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取股票日线数据（优先使用本地数据）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            force_refresh: 是否强制刷新
            
        Returns:
            股票日线数据
        """
        filepath = self.get_local_filepath(
            "stock_daily", symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # 如果不需要强制刷新，尝试加载本地数据
        if not force_refresh:
            local_data = self.load_local_data(filepath)
            if local_data is not None and self.is_data_fresh(filepath):
                self.logger.info(f"使用本地缓存数据: {symbol}")
                return local_data
        
        # 本地无数据或需要刷新，从akshare获取
        try:
            self.logger.info(f"从akshare获取数据: {symbol}")
            
            if adjust:
                stock_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                              start_date=start_date, 
                                              end_date=end_date, 
                                              adjust=adjust)
            else:
                stock_data = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                              start_date=start_date, 
                                              end_date=end_date)
            
            if not stock_data.empty:
                # 保存到本地
                self.save_local_data(stock_data, filepath)
                return stock_data
            else:
                self.logger.warning(f"akshare返回空数据: {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"从akshare获取数据失败 {symbol}: {e}")
            # 如果akshare失败，但本地有旧数据，返回旧数据
            local_data = self.load_local_data(filepath)
            if local_data is not None:
                self.logger.info(f"使用本地旧数据作为备选: {symbol}")
                return local_data
            return None
    
    def fetch_stock_info(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取股票基本信息
        """
        filepath = self.get_local_filepath("stock_info", symbol)
        
        if not force_refresh:
            local_data = self.load_local_data(filepath)
            if local_data is not None:
                return local_data
        
        try:
            self.logger.info(f"从akshare获取股票信息: {symbol}")
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            
            if not stock_info.empty:
                self.save_local_data(stock_info, filepath)
                return stock_info
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"获取股票信息失败 {symbol}: {e}")
            return self.load_local_data(filepath)
    
    def fetch_index_daily(self, symbol: str, start_date: str = "20200101", 
                         end_date: str = "20231231", 
                         force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取指数日线数据
        """
        filepath = self.get_local_filepath(
            "index_daily", symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        if not force_refresh:
            local_data = self.load_local_data(filepath)
            if local_data is not None and self.is_data_fresh(filepath):
                return local_data
        
        try:
            self.logger.info(f"从akshare获取指数数据: {symbol}")
            index_data = ak.stock_zh_index_daily(symbol=symbol)
            
            # 过滤日期范围
            if not index_data.empty:
                index_data.index = pd.to_datetime(index_data.index)
                mask = (index_data.index >= pd.to_datetime(start_date)) & \
                       (index_data.index <= pd.to_datetime(end_date))
                filtered_data = index_data[mask]
                
                if not filtered_data.empty:
                    self.save_local_data(filtered_data, filepath)
                    return filtered_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取指数数据失败 {symbol}: {e}")
            return self.load_local_data(filepath)
    
    def clear_old_data(self, max_age_days: int = 30):
        """
        清理过期数据
        
        Args:
            max_age_days: 最大保留天数
        """
        current_time = datetime.now()
        deleted_files = 0
        
        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            if os.path.isfile(filepath):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_days = (current_time - file_mtime).days
                
                if age_days > max_age_days:
                    os.remove(filepath)
                    deleted_files += 1
                    self.logger.info(f"删除过期文件: {filename}")
        
        self.logger.info(f"清理完成，共删除 {deleted_files} 个文件")

# 使用示例
def main():
    # 创建数据获取器
    fetcher = SmartDataFetcher()
    
    # 获取股票日线数据（优先使用本地缓存）
    print("=== 第一次获取数据（从akshare）===")
    data1 = fetcher.fetch_stock_daily("000001", "20230101", "20231231")
    if data1 is not None:
        print(f"获取到 {len(data1)} 条数据")
    
    print("\n=== 第二次获取相同数据（从本地缓存）===")
    data2 = fetcher.fetch_stock_daily("000001", "20230101", "20231231")
    if data2 is not None:
        print(f"获取到 {len(data2)} 条数据")
    
    print("\n=== 强制刷新数据 ===")
    data3 = fetcher.fetch_stock_daily("000001", "20230101", "20231231", force_refresh=True)
    if data3 is not None:
        print(f"获取到 {len(data3)} 条数据")
    
    # 获取股票信息
    print("\n=== 获取股票基本信息 ===")
    info = fetcher.fetch_stock_info("000001")
    if info is not None:
        print(info.head())
    
    # 清理过期数据
    fetcher.clear_old_data(max_age_days=7)

if __name__ == "__main__":
    main()