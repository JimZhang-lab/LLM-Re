"""
多线程版本的层级化实体抽取
使用 asyncio 并发处理多个样本以提升速度
"""
import asyncio
import json
import os
import sys
import time
import logging
import uuid
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# 加载 .env 配置文件
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 从环境变量读取配置
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api-inference.modelscope.cn/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
EXTRACT_LANGUAGE = os.getenv("EXTRACT_LANGUAGE", "Chinese")

# 设置环境变量（兼容旧代码）
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_MODEL"] = OPENAI_MODEL
os.environ["EXTRACT_LANGUAGE"] = EXTRACT_LANGUAGE

# 文件路径配置（支持相对路径）
INPUT_FILE = os.path.join(PROJECT_ROOT, os.getenv("INPUT_FILE", "data/zh_data_dev1.json"))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, os.getenv("OUTPUT_FILE", "output/submit_results_concurrent.json"))
TYPE_DICT_PATH = os.path.join(PROJECT_ROOT, os.getenv("TYPE_DICT_PATH", "data/coarse_fine_type_dict.json"))

# 并发配置
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))
RETRY_TIMES = int(os.getenv("RETRY_TIMES", "2"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "3"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.5"))

# 日志配置
LOG_DIR = os.path.join(PROJECT_ROOT, os.getenv("LOG_DIR", "logs"))
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))


def setup_logger(name: str = "concurrent_extractor", log_level: int = logging.INFO) -> logging.Logger:
    """
    配置日志系统（线程安全）
    
    Args:
        name: Logger名称
        log_level: 日志级别
        
    Returns:
        配置好的logger对象
    """
    # 创建logs目录
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler - 详细日志（带自动轮转）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detail_log_file = os.path.join(LOG_DIR, f'concurrent_extraction_{timestamp}.log')
    file_handler = RotatingFileHandler(
        detail_log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 错误日志文件handler
    error_log_file = os.path.join(LOG_DIR, f'concurrent_extraction_errors_{timestamp}.log')
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # logging模块本身是线程安全的，适用于asyncio和多线程环境
    logger.info(f"日志系统初始化完成")
    logger.info(f"详细日志: {detail_log_file}")
    logger.info(f"错误日志: {error_log_file}")
    
    return logger


# 创建全局logger
logger = setup_logger()

# 导入原始的 HierarchicalEntityExtractor
from test_submit import HierarchicalEntityExtractor


class ConcurrentExtractor:
    """并发实体抽取器"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_TASKS):
        """
        初始化并发抽取器
        
        Args:
            max_concurrent: 最大并发任务数
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.extractor = HierarchicalEntityExtractor()
        
        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'retried': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }


class ConcurrentExtractorWithJSONL(ConcurrentExtractor):
    """支持 JSONL 实时保存的并发抽取器（线程安全）"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_TASKS, jsonl_file: str = None):
        """
        初始化支持 JSONL 的并发抽取器
        
        Args:
            max_concurrent: 最大并发任务数
            jsonl_file: JSONL 输出文件路径
        """
        super().__init__(max_concurrent)
        self.jsonl_file = jsonl_file
        # asyncio.Lock 用于保护文件写入操作，确保线程安全
        self.file_lock = asyncio.Lock()
    
    async def process_single_sample(
        self, 
        sample: Dict[str, Any], 
        index: int, 
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        处理单个样本并立即写入 JSONL 文件
        
        Args:
            sample: 输入样本
            index: 样本索引
            retry_count: 当前重试次数
            
        Returns:
            处理结果或 None
        """
        async with self.semaphore:
            try:
                # 添加请求延迟以避免超过API限制
                if index > 0:
                    await asyncio.sleep(REQUEST_DELAY)
                
                sentence = sample['sentence']
                coarse_types = sample['coarse_types']
                
                # 记录进度
                logger.info(f"[{index + 1}/{self.stats['total']}] 开始处理: {sentence[:50]}...")
                
                # 执行实体抽取
                logger.debug(f"[{index + 1}] 调用实体抽取方法...")
                entities = await self.extractor.extract_entities_for_sentence(
                    sentence, 
                    coarse_types
                )
                logger.info(f"[{index + 1}] 抽取完成，获得 {len(entities)} 个实体")
                
                # 清理实体字段
                cleaned_entities = []
                for entity in entities:
                    cleaned_entity = {
                        'name': entity['name'],
                        'coarse_type': entity['coarse_type'],
                        'fine_type': entity['fine_type']
                    }
                    # if 'description' in entity and entity['description']:
                    #     cleaned_entity['description'] = entity['description']
                    cleaned_entities.append(cleaned_entity)
                
                # Preserve original id or generate new one
                sample_id = sample.get('id', str(uuid.uuid4()))
                
                result = {
                    "id": sample_id,
                    "sentence": sentence,
                    "entities": cleaned_entities
                }
                
                # 立即写入 JSONL 文件（使用锁保证线程安全）
                if self.jsonl_file:
                    logger.debug(f"[{index + 1}] 准备写入JSONL文件: {self.jsonl_file}")
                    try:
                        async with self.file_lock:
                            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                                json_line = json.dumps(result, ensure_ascii=False) + '\n'
                                f.write(json_line)
                                f.flush()  # 强制刷新缓冲区
                        logger.info(f"[{index + 1}] ✅ 成功写入JSONL文件")
                    except Exception as write_error:
                        logger.error(f"[{index + 1}] ❌ 写入JSONL失败: {str(write_error)}")
                        raise
                else:
                    logger.warning(f"[{index + 1}] ⚠️  未指定JSONL文件，跳过写入")
                
                self.stats['success'] += 1
                logger.info(f"[{index + 1}] ✅ 样本处理成功")
                return result
                
            except Exception as e:
                error_msg = f"样本 {index + 1} 处理失败: {str(e)}"
                logger.error(error_msg, exc_info=True)  # 添加完整的异常堆栈
                
                # 记录错误
                self.stats['errors'].append({
                    'index': index,
                    'sentence': sample.get('sentence', '')[:100],
                    'error': str(e),
                    'retry_count': retry_count
                })
                
                # 重试逻辑
                if retry_count < RETRY_TIMES:
                    logger.warning(f"重试样本 {index + 1} (尝试 {retry_count + 1}/{RETRY_TIMES})...")
                    self.stats['retried'] += 1
                    await asyncio.sleep(RETRY_DELAY)
                    return await self.process_single_sample(sample, index, retry_count + 1)
                else:
                    self.stats['failed'] += 1
                    logger.error(f"样本 {index + 1} 超过最大重试次数，放弃处理")
                    return None
    
    async def process_dataset_concurrent(
        self, 
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        并发处理整个数据集
        
        Args:
            data: 输入数据列表
            
        Returns:
            处理结果列表
        """
        self.stats['total'] = len(data)
        self.stats['start_time'] = time.time()
        
        logger.info("=" * 80)
        logger.info("🚀 开始并发处理")
        logger.info(f"📊 总样本数: {len(data)}")
        logger.info(f"⚡ 最大并发数: {self.max_concurrent}")
        logger.info("=" * 80)
        
        # 创建所有任务
        tasks = [
            self.process_single_sample(sample, i)
            for i, sample in enumerate(data)
        ]
        logger.info(f"📋 已创建 {len(tasks)} 个任务")
        
        # 并发执行所有任务
        logger.info("⏳ 开始并发执行所有任务...")
        results = await asyncio.gather(*tasks, return_exceptions=False)
        logger.info(f"✅ 所有任务执行完成，获得 {len(results)} 个结果")
        
        # 过滤掉失败的结果
        valid_results = [r for r in results if r is not None]
        logger.info(f"✅ 有效结果: {len(valid_results)}/{len(results)}")
        
        self.stats['end_time'] = time.time()
        
        return valid_results
    
    def print_statistics(self):
        """打印统计信息"""
        elapsed_time = self.stats['end_time'] - self.stats['start_time']
        
        logger.info("=" * 80)
        logger.info("📈 处理统计信息")
        logger.info("=" * 80)
        logger.info(f"✅ 成功: {self.stats['success']}/{self.stats['total']} "
                   f"({self.stats['success']/self.stats['total']*100:.1f}%)")
        logger.info(f"❌ 失败: {self.stats['failed']}/{self.stats['total']} "
                   f"({self.stats['failed']/self.stats['total']*100:.1f}%)")
        logger.info(f"🔄 重试次数: {self.stats['retried']}")
        logger.info(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"⚡ 平均速度: {self.stats['total']/elapsed_time:.2f} 样本/秒")
        logger.info("=" * 80)
        
        # 统计实体数量
        if self.stats.get('results'):
            total_entities = sum(len(r['entities']) for r in self.stats['results'])
            avg_entities = total_entities / len(self.stats['results']) if self.stats['results'] else 0
            logger.info("📊 实体统计:")
            logger.info(f"   总实体数: {total_entities}")
            logger.info(f"   平均每句: {avg_entities:.2f} 个实体")
            logger.info("=" * 80)
        
        # 显示错误详情
        if self.stats['errors']:
            logger.warning("❌ 错误详情:")
            for error in self.stats['errors'][:10]:  # 只显示前10个
                logger.warning(f"   样本 {error['index'] + 1}: {error['error']}")
            if len(self.stats['errors']) > 10:
                logger.warning(f"   ... 还有 {len(self.stats['errors']) - 10} 个错误")
            logger.info("=" * 80)


def jsonl_to_json(jsonl_file: str, json_file: str, indent: int = 4):
    """
    Convert JSON Lines file to standard JSON format.
    
    Args:
        jsonl_file: Path to input JSONL file
        json_file: Path to output JSON file
        indent: Indentation level for pretty printing
    """
    logger.info(f"🔄 转换 JSONL 到 JSON: {jsonl_file} -> {json_file}")
    
    # Check if file exists
    if not os.path.exists(jsonl_file):
        logger.error(f"❌ JSONL 文件不存在: {jsonl_file}")
        return 0
    
    # Check file size
    file_size = os.path.getsize(jsonl_file)
    logger.info(f"📊 JSONL 文件大小: {file_size} 字节")
    
    results = []
    line_count = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"❌ 第 {line_num} 行 JSON 解析失败: {e}")
                    logger.error(f"   内容: {line[:100]}...")
    
    logger.info(f"📊 JSONL 文件共 {line_count} 行，有效数据 {len(results)} 条")
    
    # Save as standard JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"✅ 转换完成，共 {len(results)} 条记录")
    return len(results)


async def process_dataset_file(
    input_file: str,
    output_file: str,
    max_samples: Optional[int] = None,
    max_concurrent: int = MAX_CONCURRENT_TASKS
):
    """
    处理数据集文件（使用 JSON Lines 格式逐条保存）
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（最终 JSON 格式）
        max_samples: 最大处理样本数（None表示处理全部）
        max_concurrent: 最大并发数
    """
    logger.info(f"📂 加载数据集: {input_file}")
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"✅ 成功加载 {len(data)} 个样本")
    
    # 限制样本数
    if max_samples:
        data = data[:max_samples]
        logger.info(f"📊 限制处理前 {max_samples} 个样本")
    
    # JSONL temporary file
    jsonl_file = output_file.replace('.json', '.jsonl')
    logger.info(f"📝 JSONL 临时文件: {jsonl_file}")
    
    # Clear/create JSONL file (overwrite mode at start)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        pass  # Create empty file
    logger.info(f"✅ JSONL 文件已创建并清空")
    
    # 创建并发抽取器（修改为支持 JSONL）
    extractor = ConcurrentExtractorWithJSONL(max_concurrent=max_concurrent, jsonl_file=jsonl_file)
    logger.info(f"✅ 抽取器已创建，JSONL文件路径: {extractor.jsonl_file}")
    
    # 并发处理（结果直接写入 JSONL）
    results = await extractor.process_dataset_concurrent(data)
    
    logger.info(f"✅ JSONL 文件已保存: {jsonl_file}")
    logger.info(f"📊 共处理 {len(results)} 条记录")
    
    # Convert JSONL to standard JSON
    logger.info(f"\n🔄 转换为标准 JSON 格式...")
    total_records = jsonl_to_json(jsonl_file, output_file, indent=4)
    
    logger.info(f"✅ 最终结果保存到: {output_file}")
    
    # 保存统计信息
    extractor.stats['results'] = results
    extractor.print_statistics()
    
    # 保存错误日志
    if extractor.stats['errors']:
        error_log_file = output_file.replace('.json', '_errors.json')
        with open(error_log_file, 'w', encoding='utf-8') as f:
            json.dump(extractor.stats['errors'], f, ensure_ascii=False, indent=2)
        logger.warning(f"📋 错误日志已保存到: {error_log_file}")
    
    logger.info("✨ 处理完成!")


async def main():
    """主函数"""
    # 处理前3个样本测试
    await process_dataset_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        max_samples=10,  # 测试：只处理前3个样本
        max_concurrent=MAX_CONCURRENT_TASKS
    )


async def process_full_dataset():
    """处理完整数据集"""
    logger.warning("⚠️  警告: 即将处理完整数据集，这可能需要较长时间...")
    response = input("是否继续? (y/N): ")
    
    if response.lower() == 'y':
        logger.info("开始处理完整数据集...")
        await process_dataset_file(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            max_samples=None,  # 处理全部
            max_concurrent=MAX_CONCURRENT_TASKS
        )
    else:
        logger.info("❌ 用户取消处理")


if __name__ == "__main__":
    # 可以根据需要选择运行模式
    
    # 模式1: 测试模式（处理前N个样本）
    asyncio.run(main())
    
    # 模式2: 完整处理（取消下面的注释）
    # asyncio.run(process_full_dataset())

