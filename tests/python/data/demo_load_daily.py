import akshare as ak
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict, Any
import re

class AkshareParquetCache:
    """基于Parquet + 时间分区的akshare数据缓存系统"""
    
    def __init__(self, cache_dir: str = "akshare_cache", partition_by: str = "month"):
        """
        Args:
            cache_dir: 缓存目录
            partition_by: 分区粒度 - day, month, year
        """
        self.cache_dir = Path(cache_dir)
        self.partition_by = partition_by
        self.setup_logging()
        self.ensure_cache_dir()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(exist_ok=True)
        self.logger.info(f"缓存目录: {self.cache_dir}")
    
    def _sanitize_symbol(self, symbol: str) -> str:
        """清理股票代码，使其适合作为文件名"""
        return re.sub(r'[^\w\-_]', '_', symbol)
    
    def _get_partition_key(self, date: datetime) -> str:
        """根据日期生成分区键"""
        if self.partition_by == "day":
            return date.strftime('%Y%m%d')
        elif self.partition_by == "month":
            return date.strftime('%Y%m')
        else:  # year
            return date.strftime('%Y')
    
    def _get_partition_path(self, data_type: str, symbol: str, partition_key: str) -> Path:
        """获取分区文件路径"""
        sanitized_symbol = self._sanitize_symbol(symbol)
        partition_dir = self.cache_dir / data_type / sanitized_symbol / f"partition={partition_key}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        return partition_dir / "data.parquet"
    
    def _get_partitions_in_range(self, start_date: str, end_date: str) -> List[str]:
        """获取时间范围内的所有分区"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        partitions = []
        current = start_dt
        
        if self.partition_by == "day":
            while current <= end_dt:
                partitions.append(current.strftime('%Y%m%d'))
                current += timedelta(days=1)
        elif self.partition_by == "month":
            while current <= end_dt:
                partitions.append(current.strftime('%Y%m'))
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        else:  # year
            year = start_dt.year
            while year <= end_dt.year:
                partitions.append(str(year))
                year += 1
        
        return partitions
    
    def _ensure_date_column(self, data: pd.DataFrame, date_column: str = "日期") -> pd.DataFrame:
        """确保数据包含日期列"""
        if date_column not in data.columns:
            raise ValueError(f"数据必须包含 {date_column} 列")
        
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        return data
    
    def _save_to_partition(self, data: pd.DataFrame, data_type: str, symbol: str, 
                          date_column: str = "日期") -> int:
        """保存数据到分区，返回新增记录数"""
        data = self._ensure_date_column(data, date_column)
        total_new_records = 0
        
        # 按分区处理数据
        data['partition'] = data[date_column].apply(self._get_partition_key)
        
        for partition_key, partition_data in data.groupby('partition'):
            filepath = self._get_partition_path(data_type, symbol, partition_key)
            
            if filepath.exists():
                # 读取现有数据并合并（去重）
                existing_data = pd.read_parquet(filepath)
                
                # 确定唯一键（通常是日期）
                unique_columns = [date_column]
                if '代码' in partition_data.columns:
                    unique_columns.append('代码')
                elif 'symbol' in partition_data.columns:
                    unique_columns.append('symbol')
                
                # 合并数据，新数据覆盖旧数据
                combined_data = pd.concat([existing_data, partition_data])
                combined_data = combined_data.drop_duplicates(subset=unique_columns, keep='last')
                combined_data = combined_data.sort_values(date_column)
                
                new_records = len(combined_data) - len(existing_data)
            else:
                combined_data = partition_data.sort_values(date_column)
                new_records = len(combined_data)
            
            # 保存数据
            combined_data.to_parquet(filepath, index=False)
            total_new_records += new_records
            
            self.logger.debug(f"分区 {partition_key} 保存完成，记录数: {len(combined_data)}")
        
        return total_new_records

    # 股票数据相关方法
    def get_stock_daily(self, symbol: str, start_date: str, end_date: str, 
                       adjust: str = "", force_update: bool = False) -> pd.DataFrame:
        """获取股票日线数据"""
        data_type = f"stock_daily_{adjust}" if adjust else "stock_daily"
        
        # 尝试从缓存获取
        if not force_update:
            cached_data = self._query_cached_data(data_type, symbol, start_date, end_date)
            if not cached_data.empty:
                self.logger.info(f"从缓存获取 {symbol} 数据: {len(cached_data)} 条记录")
                return cached_data
        
        # 从akshare获取数据
        self.logger.info(f"从akshare下载 {symbol} 数据: {start_date} 到 {end_date}")
        try:
            if adjust:
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=start_date, end_date=end_date, adjust=adjust
                )
            else:
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily",
                    start_date=start_date, end_date=end_date
                )
            
            if stock_data.empty:
                self.logger.warning(f"akshare返回空数据: {symbol}")
                return pd.DataFrame()
            
            # 添加代码列用于去重
            stock_data['代码'] = symbol
            
            # 保存到缓存
            new_records = self._save_to_partition(stock_data, data_type, symbol)
            self.logger.info(f"新增 {new_records} 条记录到缓存")
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败 {symbol}: {e}")
            # 如果akshare失败，尝试返回缓存数据
            cached_data = self._query_cached_data(data_type, symbol, start_date, end_date)
            if not cached_data.empty:
                self.logger.info(f"使用缓存数据作为备选: {symbol}")
                return cached_data
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str, force_update: bool = False) -> pd.DataFrame:
        """获取股票基本信息"""
        data_type = "stock_info"
        
        if not force_update:
            # 基本信息通常不按时间分区
            filepath = self.cache_dir / data_type / f"{self._sanitize_symbol(symbol)}.parquet"
            if filepath.exists():
                self.logger.info(f"从缓存获取 {symbol} 基本信息")
                return pd.read_parquet(filepath)
        
        try:
            self.logger.info(f"从akshare下载 {symbol} 基本信息")
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            
            if not stock_info.empty:
                # 保存基本信息
                filepath = self.cache_dir / data_type / f"{self._sanitize_symbol(symbol)}.parquet"
                filepath.parent.mkdir(parents=True, exist_ok=True)
                stock_info.to_parquet(filepath, index=False)
            
            return stock_info
            
        except Exception as e:
            self.logger.error(f"获取股票信息失败 {symbol}: {e}")
            filepath = self.cache_dir / data_type / f"{self._sanitize_symbol(symbol)}.parquet"
            if filepath.exists():
                return pd.read_parquet(filepath)
            return pd.DataFrame()
    
    def get_index_daily(self, symbol: str, start_date: str, end_date: str, 
                       force_update: bool = False) -> pd.DataFrame:
        """获取指数日线数据"""
        data_type = "index_daily"
        
        if not force_update:
            cached_data = self._query_cached_data(data_type, symbol, start_date, end_date)
            if not cached_data.empty:
                self.logger.info(f"从缓存获取指数 {symbol} 数据: {len(cached_data)} 条记录")
                return cached_data
        
        try:
            self.logger.info(f"从akshare下载指数 {symbol} 数据")
            index_data = ak.stock_zh_index_daily(symbol=symbol)
            
            if not index_data.empty:
                # 过滤日期范围
                index_data.index = pd.to_datetime(index_data.index)
                mask = (index_data.index >= pd.to_datetime(start_date)) & \
                       (index_data.index <= pd.to_datetime(end_date))
                filtered_data = index_data[mask].reset_index()
                filtered_data = filtered_data.rename(columns={'index': '日期'})
                filtered_data['代码'] = symbol
                
                # 保存到缓存
                new_records = self._save_to_partition(filtered_data, data_type, symbol, "日期")
                self.logger.info(f"新增 {new_records} 条指数记录到缓存")
                
                return filtered_data
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"获取指数数据失败 {symbol}: {e}")
            cached_data = self._query_cached_data(data_type, symbol, start_date, end_date)
            if not cached_data.empty:
                self.logger.info(f"使用缓存指数数据作为备选: {symbol}")
                return cached_data
            return pd.DataFrame()
    
    def _query_cached_data(self, data_type: str, symbol: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """查询缓存数据"""
        partitions = self._get_partitions_in_range(start_date, end_date)
        result_data = []
        
        for partition in partitions:
            filepath = self._get_partition_path(data_type, symbol, partition)
            if filepath.exists():
                try:
                    partition_data = pd.read_parquet(filepath)
                    partition_data['日期'] = pd.to_datetime(partition_data['日期'])
                    
                    # 过滤日期范围
                    mask = (partition_data['日期'] >= pd.to_datetime(start_date)) & \
                           (partition_data['日期'] <= pd.to_datetime(end_date))
                    filtered_data = partition_data[mask]
                    
                    if not filtered_data.empty:
                        result_data.append(filtered_data)
                        
                except Exception as e:
                    self.logger.warning(f"读取缓存文件失败 {filepath}: {e}")
        
        if result_data:
            combined_data = pd.concat(result_data, ignore_index=True)
            combined_data = combined_data.drop_duplicates().sort_values('日期')
            return combined_data
        
        return pd.DataFrame()
    
    def get_available_dates(self, data_type: str, symbol: str) -> Dict[str, Any]:
        """获取缓存中可用的日期范围"""
        sanitized_symbol = self._sanitize_symbol(symbol)
        data_dir = self.cache_dir / data_type / sanitized_symbol
        
        if not data_dir.exists():
            return {"available": False}
        
        all_dates = []
        for partition_dir in data_dir.iterdir():
            if partition_dir.is_dir() and partition_dir.name.startswith("partition="):
                filepath = partition_dir / "data.parquet"
                if filepath.exists():
                    try:
                        data = pd.read_parquet(filepath)
                        if '日期' in data.columns:
                            data_dates = pd.to_datetime(data['日期'])
                            all_dates.extend(data_dates)
                    except Exception as e:
                        self.logger.warning(f"读取文件失败 {filepath}: {e}")
        
        if not all_dates:
            return {"available": False}
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        return {
            "available": True,
            "min_date": min_date.strftime('%Y-%m-%d'),
            "max_date": max_date.strftime('%Y-%m-%d'),
            "total_records": len(all_dates)
        }
    
    def clear_cache(self, data_type: str = None, symbol: str = None):
        """清理缓存"""
        if data_type and symbol:
            # 清理特定股票的数据
            sanitized_symbol = self._sanitize_symbol(symbol)
            target_dir = self.cache_dir / data_type / sanitized_symbol
            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir)
                self.logger.info(f"已清理 {data_type}/{symbol} 缓存")
        elif data_type:
            # 清理特定数据类型的所有缓存
            target_dir = self.cache_dir / data_type
            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir)
                self.logger.info(f"已清理 {data_type} 所有缓存")
        else:
            # 清理所有缓存
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.logger.info("已清理所有缓存")
    
    def update_daily(self, symbols: List[str], adjust: str = ""):
        """每日增量更新"""
        today = datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        updated_symbols = []
        for symbol in symbols:
            try:
                self.logger.info(f"增量更新 {symbol}")
                data = self.get_stock_daily(symbol, yesterday, today, adjust)
                if not data.empty:
                    updated_symbols.append(symbol)
            except Exception as e:
                self.logger.error(f"更新 {symbol} 失败: {e}")
        
        self.logger.info(f"每日更新完成，成功更新 {len(updated_symbols)} 只股票")
        return updated_symbols

# 使用示例
def main():
    # 初始化缓存系统
    cache = AkshareParquetCache(partition_by="month")
    
    # 示例1: 获取股票日线数据
    print("=== 获取股票日线数据 ===")
    data1 = cache.get_stock_daily("000001", "20240101", "20240115")
    print(f"获取到 {len(data1)} 条记录")
    
    # 示例2: 再次获取相同数据（从缓存）
    print("\n=== 再次获取相同数据（从缓存）===")
    data2 = cache.get_stock_daily("000001", "20240101", "20240115")
    print(f"从缓存获取 {len(data2)} 条记录")
    
    # 示例3: 获取股票基本信息
    print("\n=== 获取股票基本信息 ===")
    info = cache.get_stock_info("000001")
    print(f"股票信息列: {list(info.columns)}")
    
    # 示例4: 获取指数数据
    print("\n=== 获取指数数据 ===")
    index_data = cache.get_index_daily("000300", "20240101", "20240115")
    print(f"获取到指数数据 {len(index_data)} 条记录")
    
    # 示例5: 查看缓存状态
    print("\n=== 缓存状态 ===")
    status = cache.get_available_dates("stock_daily", "000001")
    print(f"缓存状态: {status}")
    
    # 示例6: 每日批量更新
    print("\n=== 每日批量更新 ===")
    symbols = ["000001", "000002", "000858"]
    updated = cache.update_daily(symbols)
    print(f"更新了 {len(updated)} 只股票")

    print( data1.head() )

if __name__ == "__main__":
    main()