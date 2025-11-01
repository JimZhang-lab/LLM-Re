"""
Test submission for hierarchical entity extraction on the dataset.
Processes en_data_test1.json and outputs results in submit_example.json format.
"""
import asyncio
import json
import os
import sys
import logging
import uuid
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["OPENAI_API_BASE"] = "https://api-inference.modelscope.cn/v1"
os.environ["OPENAI_API_KEY"] = "ms-077e268d-aca4-4e06-a693-53ca0b70c132"
os.environ["OPENAI_MODEL"] = "Qwen/Qwen3-235B-A22B-Instruct-2507"
os.environ["EXTRACT_LANGUAGE"] = "Chinese"

INPUT_FILE = "/Users/jim/Desktop/extensiveWork/project/relation_extraction/RE/data/zh_data_test1.json"
# INPUT_FILE = "/Users/jim/Desktop/extensiveWork/project/relation_extraction/RE/data/en_data_test1.json"
OUTPUT_FILE = "/Users/jim/Desktop/extensiveWork/project/relation_extraction/RE/output/submit_results00t.json"
TYPE_DICT_PATH = "/Users/jim/Desktop/extensiveWork/project/relation_extraction/RE/data/coarse_fine_type_dict.json"

EXTRACT_LANGUAGE = "Chinese"

# 日志配置
LOG_DIR = "/Users/jim/Desktop/extensiveWork/project/relation_extraction/RE/logs"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5


def setup_logger(name: str = "entity_extractor", log_level: int = logging.INFO) -> logging.Logger:
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
    
    # 文件handler - 详细日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detail_log_file = os.path.join(LOG_DIR, f'extraction_{timestamp}.log')
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
    error_log_file = os.path.join(LOG_DIR, f'extraction_errors_{timestamp}.log')
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    logger.info("日志系统初始化完成")
    logger.info(f"详细日志: {detail_log_file}")
    logger.info(f"错误日志: {error_log_file}")
    
    return logger


# 创建全局logger
logger = setup_logger()

from common.openai import OpenAIRE
from core.pipeline import extract_from_text
from core.type_manager import get_type_manager, initialize_type_manager
from core.hierarchical_prompts import HIERARCHICAL_PROMPTS


class HierarchicalEntityExtractor:
    """Extractor for hierarchical entity types with coarse-fine mapping."""
    
    def __init__(self, model_name: str = os.environ["OPENAI_MODEL"]):
        """Initialize the extractor with LLM and type manager."""
        self.llm = OpenAIRE(
            model=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"]
        )
        # Initialize type manager with explicit path to coarse_fine_type_dict.json
        self.type_manager = initialize_type_manager(TYPE_DICT_PATH)
        
        # Verify that the mapping file was loaded correctly
        self._verify_type_mapping_loaded()
        self.language = os.environ["EXTRACT_LANGUAGE"] if "EXTRACT_LANGUAGE" in os.environ else EXTRACT_LANGUAGE
        
    def _verify_type_mapping_loaded(self):
        """Verify that the coarse_fine_type_dict.json file was loaded correctly."""
        if not self.type_manager.coarse_fine_mapping:
            raise FileNotFoundError("Failed to load coarse_fine_type_dict.json file")
        
        logger.info("✅ 成功加载类型映射文件:")
        logger.info(f"   - 粗粒度类型数量: {len(self.type_manager.coarse_fine_mapping)}")
        logger.info(f"   - 总类型数量: {len(self.type_manager.all_types)}")
        
        # Show some examples
        sample_coarse_types = list(self.type_manager.coarse_fine_mapping.keys())[:5]
        logger.info(f"   - 示例粗粒度类型: {sample_coarse_types}")
        
        for coarse_type in sample_coarse_types:
            fine_types = self.type_manager.coarse_fine_mapping[coarse_type]
            logger.info(f"     {coarse_type}: {len(fine_types)} 个细粒度类型")
    
    async def llm_func(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        """Wrapper for LLM function."""
        return await self.llm.complete(prompt, system_prompt, history_messages, **kwargs)
    
    def get_types_for_coarse_types(self, coarse_types: List[str]) -> str:
        """
        Get fine-grained types for the given coarse types.
        Only include coarse types that have mappings in the dictionary.
        
        Args:
            coarse_types: List of coarse types from the dataset
            
        Returns:
            Formatted string of fine types for prompt
        """
        fine_types = []
        valid_coarse_types = []
        
        for coarse_type in coarse_types:
            if coarse_type in self.type_manager.coarse_fine_mapping:
                # Get all fine types for this coarse type
                fine_types.extend(self.type_manager.coarse_fine_mapping[coarse_type])
                valid_coarse_types.append(coarse_type)
            else:
                # Skip coarse types that don't have mappings
                logger.warning(f"跳过未找到映射的粗粒度类型: {coarse_type}")
        
        # Remove duplicates (no limit on types)
        unique_types = list(set(fine_types))
        
        logger.info(f"✅ 使用有效粗粒度类型: {valid_coarse_types}")
        logger.info(f"📋 对应的细粒度类型数量: {len(unique_types)}")
        
        return ", ".join(sorted(unique_types))
    
    def get_coarse_fine_mapping_for_types(self, coarse_types: List[str]) -> Dict[str, List[str]]:
        """
        Get the coarse-fine mapping for the given coarse types.
        Only include coarse types that have mappings with non-empty fine types in the dictionary.
        
        Args:
            coarse_types: List of coarse types from the dataset
            
        Returns:
            Dictionary mapping coarse types to their fine types (only valid ones with non-empty lists)
        """
        mapping = {}
        skipped_types = []
        
        for coarse_type in coarse_types:
            if coarse_type in self.type_manager.coarse_fine_mapping:
                fine_types = self.type_manager.coarse_fine_mapping[coarse_type]
                # Only include if fine_types is not empty
                if fine_types and len(fine_types) > 0:
                    mapping[coarse_type] = fine_types
                else:
                    skipped_types.append(f"{coarse_type} (无细粒度类型)")
            else:
                skipped_types.append(f"{coarse_type} (未找到映射)")
        
        # Log skipped types if any
        if skipped_types:
            logger.warning(f"  跳过的类型: {', '.join(skipped_types)}")
        
        return mapping
    
    async def extract_entities_for_sentence(self, sentence: str, coarse_types: List[str]) -> List[Dict[str, str]]:
        """
        Three-stage extraction with forward and reverse approaches:
        1. Forward: Extract coarse types first, then fine types
        2. Reverse: Extract fine types first, then map to coarse types
        3. Merge: Compare and merge results from both approaches
        
        Args:
            sentence: Input sentence
            coarse_types: Available coarse types for this sentence
            
        Returns:
            List of entities with both coarse and fine types (merged from both approaches)
        """
        # Get coarse-fine mapping for this sentence (only includes types with fine types)
        coarse_fine_mapping = self.get_coarse_fine_mapping_for_types(coarse_types)
        
        if not coarse_fine_mapping:
            logger.warning(f"  没有可用的类型映射（所有类型都没有细粒度类型），跳过抽取")
            return []
        
        # Get valid coarse types (those that have fine types)
        valid_coarse_types = list(coarse_fine_mapping.keys())
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📖 处理句子: {sentence[:100]}...")
        logger.info(f"🎯 原始粗粒度类型: {coarse_types}")
        if len(valid_coarse_types) < len(coarse_types):
            logger.warning(f"  过滤后剩余 {len(valid_coarse_types)}/{len(coarse_types)} 个有效类型")
        logger.info(f"✅ 有效粗粒度类型（有细粒度映射）: {valid_coarse_types}")
        
        try:
            # ============ Stage 1: Forward Extraction (Coarse -> Fine) ============
            logger.info(f"\n【阶段1-正向】先抽取粗粒度，再抽取细粒度...")
            forward_entities = await self._forward_extraction(sentence, valid_coarse_types, coarse_fine_mapping)
            
            # ============ Stage 2: Reverse Extraction (Fine -> Coarse) ============
            logger.info(f"\n【阶段2-反向】先抽取细粒度，再映射粗粒度...")
            reverse_entities = await self._reverse_extraction(sentence, coarse_fine_mapping)
            
            # ============ Stage 3: Merge Results ============
            logger.info(f"\n【阶段3-合并】对比并合并两种方法的结果...")
            merged_entities = self._merge_extraction_results(forward_entities, reverse_entities, sentence)
            
            logger.info(f"\n✅ 最终合并后提取到 {len(merged_entities)} 个完整实体")
            for entity in merged_entities:
                source = entity.get('source', 'unknown')
                logger.info(f"   - {entity['name']}: {entity['coarse_type']} -> {entity['fine_type']} (来源: {source})")
            
            return merged_entities
            
        except Exception as e:
            logger.error(f" 提取失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _forward_extraction(self, sentence: str, valid_coarse_types: List[str], coarse_fine_mapping: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Forward extraction: Coarse -> Fine
        
        Args:
            sentence: Input sentence
            valid_coarse_types: List of valid coarse types
            coarse_fine_mapping: Mapping from coarse to fine types
            
        Returns:
            List of entities with coarse and fine types
        """
        logger.info(f"  ➡️  正向抽取：粗粒度 → 细粒度")
        
        # Extract coarse entities
        coarse_entities = await self._extract_coarse_entities(sentence, valid_coarse_types)
        
        if not coarse_entities:
            logger.warning(f"    未提取到实体（正向）")
            return []
        
        logger.info(f"  ✅ 正向提取到 {len(coarse_entities)} 个粗粒度实体")
        for entity in coarse_entities:
            logger.info(f"     - {entity['name']} -> {entity['coarse_type']}")
        
        # Extract fine types for each entity
        result_entities = await self._extract_fine_types_for_entities(
            sentence, coarse_entities, coarse_fine_mapping
        )
        
        # Mark source as forward
        for entity in result_entities:
            entity['source'] = '正向'
        
        logger.info(f"  ✅ 正向完成：{len(result_entities)} 个完整实体")
        return result_entities
    
    async def _reverse_extraction(self, sentence: str, coarse_fine_mapping: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        Reverse extraction: Fine -> Coarse
        直接使用映射表将细粒度类型映射到粗粒度类型
        
        Args:
            sentence: Input sentence
            coarse_fine_mapping: Mapping from coarse to fine types
            
        Returns:
            List of entities with fine and coarse types
        """
        logger.info(f"  ⬅️  反向抽取：细粒度 → 粗粒度（直接映射）")
        
        # Build reverse mapping: fine_type -> coarse_type for fast lookup
        fine_to_coarse = {}
        all_fine_types = []
        
        for coarse_type, fine_types in coarse_fine_mapping.items():
            for fine_type in fine_types:
                fine_type_lower = fine_type.lower()
                # Store the mapping (lowercase for case-insensitive lookup)
                fine_to_coarse[fine_type_lower] = coarse_type
                # Keep original case for extraction
                if fine_type_lower not in [ft.lower() for ft in all_fine_types]:
                    all_fine_types.append(fine_type)
        
        logger.info(f"  📋 可用细粒度类型数量: {len(all_fine_types)}")
        logger.info(f"  📊 构建反向映射表: {len(fine_to_coarse)} 个细粒度→粗粒度映射")
        
        # Extract entities with fine types directly
        fine_entities = await self._extract_fine_entities_directly(sentence, all_fine_types)
        
        if not fine_entities:
            logger.warning(f"    未提取到实体（反向）")
            return []
        
        logger.info(f"  ✅ 反向提取到 {len(fine_entities)} 个细粒度实体")
        for entity in fine_entities:
            logger.info(f"     - {entity['name']} -> {entity['fine_type']}")
        
        # Use the reverse mapping to find coarse types (EXACT MATCH ONLY)
        result_entities = []
        for entity in fine_entities:
            fine_type = entity['fine_type']
            fine_type_lower = fine_type.lower()
            
            # Direct lookup in reverse mapping - EXACT MATCH ONLY
            coarse_type = fine_to_coarse.get(fine_type_lower)
            
            if coarse_type:
                result_entities.append({
                    'name': entity['name'],
                    'coarse_type': coarse_type,
                    'fine_type': fine_type,
                    'description': entity.get('description', ''),
                    'source': '反向'
                })
                logger.info(f"     ✓ {entity['name']}: {fine_type} → {coarse_type} (精确匹配)")
            else:
                logger.info(f"     ✗ {entity['name']}: {fine_type} 无法精确映射到粗粒度类型，跳过")
        
        logger.info(f"  ✅ 反向完成：{len(result_entities)} 个完整实体（精确匹配）")
        return result_entities
    
    def _merge_extraction_results(self, forward_entities: List[Dict], reverse_entities: List[Dict], sentence: str) -> List[Dict]:
        """
        Merge results from forward and reverse extraction.
        
        Strategy:
        1. If entity appears in both, prefer the one with more specific fine type
        2. If entity only appears in one, include it
        3. Deduplicate by entity name (case-insensitive)
        
        Args:
            forward_entities: Results from forward extraction
            reverse_entities: Results from reverse extraction
            sentence: Original sentence for reference
            
        Returns:
            Merged list of entities
        """
        logger.info(f"  🔀 合并策略：优先选择更具体的细粒度类型")
        logger.info(f"  📊 正向结果: {len(forward_entities)} 个实体")
        logger.info(f"  📊 反向结果: {len(reverse_entities)} 个实体")
        
        # Build a dictionary keyed by lowercase entity name
        entity_dict = {}
        
        # Add forward entities
        for entity in forward_entities:
            name_lower = entity['name'].lower()
            entity_dict[name_lower] = entity.copy()
        
        # Process reverse entities
        for entity in reverse_entities:
            name_lower = entity['name'].lower()
            
            if name_lower in entity_dict:
                # Entity exists in both - compare and choose better one
                existing = entity_dict[name_lower]
                
                # Prefer entity where coarse and fine types are different
                existing_same = existing['coarse_type'].lower() == existing['fine_type'].lower()
                new_same = entity['coarse_type'].lower() == entity['fine_type'].lower()
                
                if existing_same and not new_same:
                    # New entity has different types, prefer it
                    logger.info(f"  🔄 替换 '{entity['name']}': {existing['source']}结果粗细类型相同，使用{entity['source']}结果")
                    entity_dict[name_lower] = entity.copy()
                    entity_dict[name_lower]['source'] = '反向(替换)'
                elif not existing_same and new_same:
                    # Existing entity is better, keep it
                    logger.info(f"  ✓ 保留 '{entity['name']}': {existing['source']}结果更具体")
                else:
                    # Both are similar, mark as from both sources
                    logger.info(f"  ≈ 重复 '{entity['name']}': 两种方法结果相似，保留{existing['source']}结果")
                    entity_dict[name_lower]['source'] = '正向+反向'
            else:
                # New entity only in reverse
                logger.info(f"  ➕ 新增 '{entity['name']}': 仅在反向抽取中发现")
                entity_dict[name_lower] = entity.copy()
        
        # Convert back to list
        merged_entities = list(entity_dict.values())
        
        logger.info(f"  ✅ 合并完成：总计 {len(merged_entities)} 个唯一实体")
        logger.info(f"     - 正向独有: {sum(1 for e in merged_entities if e['source'] == '正向')}")
        logger.info(f"     - 反向独有: {sum(1 for e in merged_entities if e['source'] == '反向')}")
        logger.info(f"     - 反向替换: {sum(1 for e in merged_entities if e['source'] == '反向(替换)')}")
        logger.info(f"     - 两者都有: {sum(1 for e in merged_entities if e['source'] == '正向+反向')}")
        
        return merged_entities
    
    async def _extract_coarse_entities(self, sentence: str, coarse_types: List[str], max_gleaning: int = 1) -> List[Dict]:
        """
        Stage 1: Extract entities and classify them into coarse types with gleaning.
        
        Args:
            sentence: The text to extract entities from
            coarse_types: List of available coarse types
            max_gleaning: Maximum number of gleaning iterations to find missed entities
            
        Returns:
            List of entities with name and coarse_type
        """
        from core.prompts import PROMPTS
        
        # Get delimiters from prompts
        tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        
        # Add "other/其他" type for misannotated data
        coarse_types_with_other = coarse_types + ["other", "其他"]
        
        # Build coarse extraction prompt
        coarse_types_str = ", ".join(coarse_types_with_other)
        
        # Select examples based on language environment variable
        if self.language.lower() == "chinese":
            examples = HIERARCHICAL_PROMPTS.get("coarse_extraction_examples_zh", 
                                               HIERARCHICAL_PROMPTS["coarse_extraction_examples_en"])
        else:
            examples = HIERARCHICAL_PROMPTS["coarse_extraction_examples_en"]
        examples_str = "\n".join(examples)
        
        # Initial extraction
        system_prompt = HIERARCHICAL_PROMPTS["coarse_extraction_system_prompt"].format(
            entity_types=coarse_types_str,
            tuple_delimiter=tuple_delimiter,
            completion_delimiter=completion_delimiter,
            language=self.language,
            examples=examples_str,
            input_text=sentence
        )
        
        # Call LLM
        response = await self.llm_func(system_prompt)
        
        # Parse response
        entities = self._parse_coarse_entities_response(response, coarse_types_with_other, tuple_delimiter, completion_delimiter)
        
        # Gleaning: Continue extraction for missed entities
        for gleaning_round in range(max_gleaning):
            if not entities:
                break
                
            logger.info(f"    🔄 继续抽取（第 {gleaning_round + 1}/{max_gleaning} 轮）检查遗漏实体...")
            
            # Build continue extraction prompt
            extracted_names = [e["name"] for e in entities]
            continue_prompt = self._build_coarse_continue_prompt(
                sentence, 
                coarse_types_str, 
                extracted_names,
                tuple_delimiter,
                completion_delimiter
            )
            
            # Call LLM for continue extraction
            continue_response = await self.llm_func(continue_prompt)
            
            # Parse continue response
            new_entities = self._parse_coarse_entities_response(
                continue_response, 
                coarse_types_with_other, 
                tuple_delimiter, 
                completion_delimiter
            )
            
            # Add new entities (avoid duplicates)
            existing_names_lower = {e["name"].lower() for e in entities}
            added_count = 0
            for entity in new_entities:
                if entity["name"].lower() not in existing_names_lower:
                    entities.append(entity)
                    existing_names_lower.add(entity["name"].lower())
                    added_count += 1
            
            if added_count > 0:
                logger.info(f"    ✅ 发现 {added_count} 个遗漏实体")
            else:
                logger.info(f"    ✓ 未发现遗漏实体，停止继续抽取")
                break
        
        return entities
    
    def _parse_coarse_entities_response(
        self, 
        response: str, 
        coarse_types: List[str], 
        tuple_delimiter: str, 
        completion_delimiter: str
    ) -> List[Dict]:
        """Parse coarse entities from LLM response."""
        entities = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line == completion_delimiter:
                continue
            
            # Parse: entity<|>name<|>type<|>description
            if tuple_delimiter in line:
                parts = line.split(tuple_delimiter)
                if len(parts) >= 3 and parts[0].strip().lower() == "entity":
                    entity_name = parts[1].strip()
                    entity_type = parts[2].strip()
                    
                    # Validate coarse type
                    if entity_type.lower() in [ct.lower() for ct in coarse_types]:
                        entities.append({
                            "name": entity_name,
                            "coarse_type": entity_type
                        })
                    else:
                        # Try to map to one of the coarse types
                        mapped_type = self._map_to_coarse_type(entity_type, coarse_types, {})
                        if mapped_type:
                            entities.append({
                                "name": entity_name,
                                "coarse_type": mapped_type
                            })
        
        return entities
    
    def _build_coarse_continue_prompt(
        self, 
        sentence: str, 
        coarse_types_str: str, 
        extracted_names: List[str],
        tuple_delimiter: str,
        completion_delimiter: str
    ) -> str:
        """Build continue extraction prompt for coarse entities."""
        extracted_list = "\n".join([f"- {name}" for name in extracted_names])
        
        prompt = f"""---Task---
You have already extracted the following entities from the text:

{extracted_list}

Now, carefully review the text again and extract ANY MISSED entities that were NOT in the above list.

---Text---
{sentence}

---Available Coarse Types---
{coarse_types_str}

---Instructions---
1. ONLY output entities that are NOT in the already extracted list above
2. Use the same format: entity{tuple_delimiter}name{tuple_delimiter}coarse_type{tuple_delimiter}description
3. If there are NO missed entities, just output: {completion_delimiter}
4. Do NOT repeat any entity from the already extracted list

<Output>
"""
        return prompt
    
    async def _extract_fine_types_for_entities(
        self, 
        sentence: str, 
        coarse_entities: List[Dict], 
        coarse_fine_mapping: Dict[str, List[str]]
    ) -> List[Dict]:
        """
        Stage 2: Extract fine-grained types for each entity based on its coarse type.
        
        Args:
            sentence: The original sentence
            coarse_entities: List of entities with coarse types
            coarse_fine_mapping: Mapping from coarse to fine types
            
        Returns:
            List of entities with both coarse_type and fine_type
        """
        result_entities = []
        
        # Group entities by coarse type for batch processing
        entities_by_coarse = defaultdict(list)
        for entity in coarse_entities:
            coarse_type = entity["coarse_type"]
            entities_by_coarse[coarse_type].append(entity)
        
        # Process each coarse type group
        for coarse_type, entities in entities_by_coarse.items():
            # Filter out entities with "other" or "其他" type (misannotated data)
            if coarse_type.lower() in ["other", "其他"]:
                logger.info(f"\n  ⚠️  跳过标注为 '{coarse_type}' 的实体 ({len(entities)} 个): 可能为标注错误")
                for entity in entities:
                    logger.info(f"     - {entity['name']} (标注错误，已过滤)")
                continue
            
            logger.info(f"\n  处理粗粒度类型: {coarse_type} ({len(entities)} 个实体)")
            
            # Get fine types for this coarse type
            fine_types = coarse_fine_mapping.get(coarse_type, [])
            
            if not fine_types:
                logger.warning(f"    没有找到细粒度类型，使用粗粒度类型")
                for entity in entities:
                    result_entities.append({
                        "name": entity["name"],
                        "coarse_type": coarse_type,
                        "fine_type": coarse_type
                    })
                continue
            
            logger.info(f"  📋 可用细粒度类型 ({len(fine_types)}): {fine_types[:10]}{'...' if len(fine_types) > 10 else ''}")
            
            # Extract fine types for all entities of this coarse type (with gleaning)
            fine_typed_entities = await self._extract_fine_types_with_gleaning(
                sentence, entities, coarse_type, fine_types, max_gleaning=1
            )
            
            # Add to results
            result_entities.extend(fine_typed_entities)
        
        return result_entities
    
    async def _extract_fine_types_with_gleaning(
        self,
        sentence: str,
        entities: List[Dict],
        coarse_type: str,
        fine_types: List[str],
        max_gleaning: int = 1
    ) -> List[Dict]:
        """
        Extract fine types for entities with gleaning to catch missed entities.
        
        Args:
            sentence: The sentence
            entities: List of entities with coarse type
            coarse_type: The coarse type
            fine_types: Available fine types
            max_gleaning: Maximum gleaning iterations
            
        Returns:
            List of entities with both coarse and fine types
        """
        result_entities = []
        
        # Initial extraction for all entities
        for entity in entities:
            entity_name = entity["name"]
            fine_type = await self._extract_fine_type_for_entity(
                sentence, entity_name, coarse_type, fine_types
            )
            
            result_entities.append({
                "name": entity_name,
                "coarse_type": coarse_type,
                "fine_type": fine_type if fine_type else coarse_type
            })
            
            logger.info(f"    ✓ {entity_name}: {coarse_type} -> {fine_type if fine_type else coarse_type}")
        
        # Gleaning: Check for missed entities of this coarse type
        for gleaning_round in range(max_gleaning):
            logger.info(f"    🔄 继续抽取（第 {gleaning_round + 1}/{max_gleaning} 轮）检查遗漏的 {coarse_type} 类型实体...")
            
            # Build continue extraction prompt for this coarse type
            extracted_names = [e["name"] for e in result_entities]
            continue_prompt = self._build_fine_continue_prompt(
                sentence,
                coarse_type,
                fine_types,
                extracted_names
            )
            
            # Call LLM for continue extraction
            continue_response = await self.llm_func(continue_prompt)
            
            # Parse new entities
            new_entity_names = self._parse_fine_continue_response(continue_response, extracted_names)
            
            if not new_entity_names:
                logger.info(f"    ✓ 未发现遗漏的 {coarse_type} 实体，停止继续抽取")
                break
            
            logger.info(f"    ✅ 发现 {len(new_entity_names)} 个遗漏的 {coarse_type} 实体")
            
            # Extract fine types for new entities
            for entity_name in new_entity_names:
                fine_type = await self._extract_fine_type_for_entity(
                    sentence, entity_name, coarse_type, fine_types
                )
                
                result_entities.append({
                    "name": entity_name,
                    "coarse_type": coarse_type,
                    "fine_type": fine_type if fine_type else coarse_type
                })
                
                logger.info(f"    ✓ (补充) {entity_name}: {coarse_type} -> {fine_type if fine_type else coarse_type}")
        
        return result_entities
    
    def _build_fine_continue_prompt(
        self,
        sentence: str,
        coarse_type: str,
        fine_types: List[str],
        extracted_names: List[str]
    ) -> str:
        """Build continue extraction prompt for fine-grained entities."""
        extracted_list = "\n".join([f"- {name}" for name in extracted_names])
        fine_types_str = ", ".join(fine_types)  # No limit on types
        
        prompt = f"""---Task---
You have already extracted the following entities of type '{coarse_type}' from the text:

{extracted_list}

Now, carefully review the text again and find ANY MISSED entities of type '{coarse_type}' that were NOT in the above list.

---Text---
{sentence}

---Coarse Type---
{coarse_type}

---Instructions---
1. ONLY look for entities of type '{coarse_type}'
2. ONLY output entities that are NOT in the already extracted list above
3. Output format: Just list the entity names, one per line
4. If there are NO missed entities of type '{coarse_type}', just output: NONE

<Output>
"""
        return prompt
    
    def _parse_fine_continue_response(self, response: str, existing_names: List[str]) -> List[str]:
        """Parse continue extraction response for fine entities."""
        response = response.strip()
        
        if not response or "NONE" in response.upper():
            return []
        
        # Parse entity names from response
        new_names = []
        existing_names_lower = {name.lower() for name in existing_names}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove leading markers like "- ", "* ", numbers, etc.
            entity_name = line.lstrip('-*•123456789. ').strip()
            
            if entity_name and entity_name.lower() not in existing_names_lower:
                new_names.append(entity_name)
                existing_names_lower.add(entity_name.lower())
        
        return new_names
    
    async def _extract_fine_type_for_entity(
        self, 
        sentence: str, 
        entity_name: str, 
        coarse_type: str, 
        fine_types: List[str],
        max_retries: int = 2
    ) -> str:
        """
        Extract fine-grained type for a single entity.
        If the extracted fine type is the same as coarse type, add error to history and retry extraction.
        
        Args:
            sentence: The sentence containing the entity
            entity_name: Name of the entity
            coarse_type: Coarse type of the entity
            fine_types: List of available fine types for this coarse type
            max_retries: Maximum number of retries if coarse and fine types are the same
            
        Returns:
            Fine-grained type or None if extraction fails
        """
        history_messages = []  # Track conversation history for retries
        
        for attempt in range(max_retries + 1):
            try:
                # Get examples for this coarse type based on language
                if self.language.lower() == "chinese":
                    examples_dict = HIERARCHICAL_PROMPTS.get("fine_extraction_examples_zh", 
                                                            HIERARCHICAL_PROMPTS["fine_extraction_examples"])
                else:
                    examples_dict = HIERARCHICAL_PROMPTS.get("fine_extraction_examples_en", 
                                                            HIERARCHICAL_PROMPTS["fine_extraction_examples"])
                
                examples = examples_dict.get(
                    coarse_type.lower(),
                    examples_dict.get("person", examples_dict.get("人", ""))
                )
                
                # Build fine extraction prompt (no limit on types)
                fine_types_str = ", ".join(fine_types)
                
                prompt = HIERARCHICAL_PROMPTS["fine_extraction_system_prompt"].format(
                    entity_name=entity_name,
                    coarse_type=coarse_type,
                    sentence=sentence,
                    fine_types=fine_types_str,
                    examples=examples
                )
                
                # Call LLM with history messages on retry
                if history_messages:
                    # When using history, the complete() function ignores the prompt parameter
                    # The new prompt should already be in the history_messages
                    response = await self.llm_func("", history_messages=history_messages)
                else:
                    response = await self.llm_func(prompt)
                
                # Parse response
                extracted_type = self._parse_type_from_response(response, fine_types)
                
                # Check if extracted type is the same as coarse type
                if extracted_type and extracted_type.lower() != coarse_type.lower():
                    # Success: fine type is different from coarse type
                    return extracted_type
                elif extracted_type and extracted_type.lower() == coarse_type.lower():
                    # Fine type is same as coarse type, need to retry
                    if attempt < max_retries:
                        logger.warning(f"      细粒度类型与粗粒度类型相同 ({extracted_type}), 将错误添加到历史并重新抽取 (尝试 {attempt + 1}/{max_retries})...")
                        
                        # Add the error to history for next attempt
                        error_message = f"""❌ ERROR: Your previous answer "{extracted_type}" is INCORRECT because it is the same as the coarse type "{coarse_type}".

🔴 CRITICAL REQUIREMENT: You MUST provide a MORE SPECIFIC fine-grained type from the available types.

Available fine-grained types: {fine_types_str}

Please provide a MORE SPECIFIC type that is DIFFERENT from "{coarse_type}". Choose the most appropriate fine-grained type from the list above."""
                        
                        # Build history messages in the format expected by OpenAI API
                        if not history_messages:
                            # First retry: add initial user message and assistant's wrong response
                            history_messages.append({"role": "user", "content": prompt})
                            history_messages.append({"role": "assistant", "content": f"Type: {extracted_type}"})
                        
                        # Add error feedback as user message
                        history_messages.append({"role": "user", "content": error_message})
                        
                        continue
                    else:
                        logger.warning(f"      多次尝试后仍为相同类型，保持粗粒度类型")
                        return coarse_type
                else:
                    # No valid type extracted
                    if attempt < max_retries:
                        logger.warning(f"      未提取到有效类型, 将错误添加到历史并重试 (尝试 {attempt + 1}/{max_retries})...")
                        
                        # Add the error to history
                        error_message = f"""❌ ERROR: Your previous response did not contain a valid type.

Available fine-grained types: {fine_types_str}

Please provide ONE of the above fine-grained types in the format: Type: [type_name]"""
                        
                        if not history_messages:
                            history_messages.append({"role": "user", "content": prompt})
                            history_messages.append({"role": "assistant", "content": response})
                        
                        history_messages.append({"role": "user", "content": error_message})
                        continue
                    else:
                        logger.warning(f"      无法提取有效细粒度类型，使用粗粒度类型")
                        return coarse_type
                
            except Exception as e:
                logger.info(f"    ❌ 提取细粒度类型失败 (尝试 {attempt + 1}): {e}")
                if attempt >= max_retries:
                    return coarse_type
        
        return coarse_type
    
    async def _extract_fine_entities_directly(self, sentence: str, fine_types: List[str], max_gleaning: int = 1) -> List[Dict]:
        """
        Extract entities directly with fine-grained types (for reverse extraction).
        
        Args:
            sentence: The text to extract entities from
            fine_types: List of all available fine-grained types
            max_gleaning: Maximum number of gleaning iterations
            
        Returns:
            List of entities with name and fine_type
        """
        try:
            # Call extract_from_text with fine types
            # extract_from_text returns (entities_dict, relationships_list)
            entities_dict, relationships_list = await extract_from_text(
                text=sentence,
                entity_types=fine_types,
                llm_func=self.llm_func,
                use_hierarchical_types=False,
                max_gleaning=max_gleaning,
                language=self.language
            )
            
            # Convert to our format
            entities = []
            for entity_name, entity_list in entities_dict.items():
                # entity_list is a list of entity dicts for this entity name
                if entity_list:
                    # Take the first entity (should only be one in most cases)
                    entity_data = entity_list[0]
                    entity_type = entity_data.get("entity_type", "")
                    description = entity_data.get("description", "")
                    
                    entities.append({
                        'name': entity_name,
                        'fine_type': entity_type,
                        'description': description
                    })
            
            return entities
            
        except Exception as e:
            logger.info(f"    ❌ 直接抽取细粒度实体失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_type_from_response(self, response: str, valid_types: List[str]) -> str:
        """
        Parse entity type from LLM response using EXACT matching only.
        Unified parser for all type extraction scenarios.
        
        Args:
            response: LLM response
            valid_types: List of valid types to match against
            
        Returns:
            Matched type or None
        """
        response = response.strip()
        
        # Try to extract from "Type: xxx" or similar patterns
        patterns = ["type:", "answer:", "类型:", "答案:"]
        extracted_text = response.lower()
        
        for pattern in patterns:
            if pattern in extracted_text:
                extracted_text = extracted_text.split(pattern)[-1].strip()
                break
        
        # Remove common punctuation
        extracted_text = extracted_text.replace(".", "").replace(",", "").replace("!", "").replace(";", "").strip()
        
        # Extract first word if multiple words
        first_word = extracted_text.split()[0] if extracted_text.split() else extracted_text
        
        # Try exact match with first word
        for valid_type in valid_types:
            if first_word == valid_type.lower():
                return valid_type
        
        # Try exact match with full extracted text
        for valid_type in valid_types:
            if extracted_text == valid_type.lower():
                return valid_type
        
        # Try exact match with original response
        response_lower = response.lower()
        for valid_type in valid_types:
            if response_lower == valid_type.lower():
                return valid_type
        
        # Try exact match with first line
        first_line = response.split('\n')[0].strip().lower()
        for valid_type in valid_types:
            if first_line == valid_type.lower():
                return valid_type
        
        return None
    
    def _map_to_coarse_type(self, extracted_type: str, available_coarse_types: List[str], coarse_fine_mapping: Dict[str, List[str]] = None) -> str:
        """
        Map an extracted fine type to one of the available coarse types using EXACT matching only.
        No fuzzy or partial matching is performed.
        
        Args:
            extracted_type: The extracted entity type
            available_coarse_types: Available coarse types for this sentence
            coarse_fine_mapping: Coarse-fine mapping for this sentence
            
        Returns:
            Mapped coarse type or None if no exact match found
        """
        extracted_type_lower = extracted_type.lower()
        
        # Check if the extracted type is already a coarse type (exact match)
        if extracted_type_lower in [ct.lower() for ct in available_coarse_types]:
            return extracted_type
        
        # Use the provided mapping if available (exact match only)
        if coarse_fine_mapping:
            for coarse_type, fine_types in coarse_fine_mapping.items():
                if extracted_type_lower in [ft.lower() for ft in fine_types]:
                    return coarse_type
        
        # Try to find a mapping through the type manager (exact match only)
        for coarse_type in available_coarse_types:
            if coarse_type in self.type_manager.coarse_fine_mapping:
                fine_types = self.type_manager.coarse_fine_mapping[coarse_type]
                if extracted_type_lower in [ft.lower() for ft in fine_types]:
                    return coarse_type
        
        # No exact match found
        return None
    
    async def _re_extract_with_fine_types(self, sentence: str, entity_name: str, coarse_type: str, coarse_fine_mapping: Dict[str, List[str]]) -> str:
        """
        Re-extract entity type using optimized prompt with only fine types.
        
        Args:
            sentence: Original sentence
            entity_name: Name of the entity to re-extract
            coarse_type: The coarse type
            coarse_fine_mapping: Mapping of coarse to fine types
            
        Returns:
            A better fine-grained type or None if no better one found
        """
        if coarse_type not in coarse_fine_mapping:
            return None
        
        fine_types = coarse_fine_mapping[coarse_type]
        if not fine_types:
            return None
        
        logger.info(f"🔄 重新抽取实体 '{entity_name}' 的细粒度类型...")
        logger.info(f"📋 可用细粒度类型 ({len(fine_types)}个): {fine_types[:10]}{'...' if len(fine_types) > 10 else ''}")
        
        try:
            # Get appropriate examples for this coarse type based on language
            if self.language.lower() == "chinese":
                examples_dict = HIERARCHICAL_PROMPTS.get("re_extraction_examples_zh", 
                                                        HIERARCHICAL_PROMPTS["re_extraction_examples"])
            else:
                examples_dict = HIERARCHICAL_PROMPTS.get("re_extraction_examples_en", 
                                                        HIERARCHICAL_PROMPTS["re_extraction_examples"])
            
            examples = examples_dict.get(
                coarse_type, 
                examples_dict.get("person", examples_dict.get("人", ""))
            )
            
            # Build re-extraction prompt (no limit on types)
            fine_types_str = ", ".join(fine_types)
            re_extraction_prompt = HIERARCHICAL_PROMPTS["re_extraction_prompt"].format(
                entity_name=entity_name,
                sentence=sentence,
                fine_types=fine_types_str,
                examples=examples
            )
            
            # Call LLM with optimized prompt
            response = await self.llm_func(re_extraction_prompt)
            
            # Extract type from response
            extracted_type = self._parse_type_from_response(response, fine_types)
            
            if extracted_type:
                logger.info(f"✅ 重新抽取成功: {entity_name} -> {coarse_type} -> {extracted_type}")
                return extracted_type
            
            # Fallback: use general extraction with fine types only
            logger.info(f"🔄 尝试使用通用抽取方法...")
            entities, relationships = await extract_from_text(
                text=sentence,
                llm_func=self.llm_func,
                entity_types=fine_types,  # Only use fine types (no limit)
                language=self.language,
                use_hierarchical_types=False,  # Disable hierarchical types
                type_mode="fine",
                max_gleaning=0,  # No gleaning for re-extraction
                max_types_for_prompt=len(fine_types)  # Use all fine types
            )
            
            # Find the entity in results
            for extracted_entity_name, entity_list in entities.items():
                if extracted_entity_name.lower() == entity_name.lower():
                    entity = entity_list[0]
                    extracted_type = entity.get('entity_type', '')
                    
                    # Validate type (exact match only)
                    if extracted_type.lower() in [ft.lower() for ft in fine_types]:
                        logger.info(f"✅ 通用抽取成功 (精确匹配): {entity_name} -> {coarse_type} -> {extracted_type}")
                        return extracted_type
            
            # No valid type found after all attempts
            logger.warning(f"  无法提取有效细粒度类型，保持原类型: {entity_name} -> {coarse_type}")
            return None
            
        except Exception as e:
            logger.error(f" 重新抽取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def show_type_mapping_stats(self, coarse_types: List[str]):
        """Show statistics about the type mapping for given coarse types."""
        logger.info(f"\n📊 类型映射统计:")
        logger.info(f"   粗粒度类型数量: {len(coarse_types)}")
        
        total_fine_types = 0
        for coarse_type in coarse_types:
            if coarse_type in self.type_manager.coarse_fine_mapping:
                fine_count = len(self.type_manager.coarse_fine_mapping[coarse_type])
                total_fine_types += fine_count
                logger.info(f"   {coarse_type}: {fine_count} 个细粒度类型")
            else:
                logger.info(f"   {coarse_type}: 未找到映射")
        
        logger.info(f"   总细粒度类型数量: {total_fine_types}")


def jsonl_to_json(jsonl_file: str, json_file: str, indent: int = 4):
    """
    Convert JSON Lines file to standard JSON format.
    
    Args:
        jsonl_file: Path to input JSONL file
        json_file: Path to output JSON file
        indent: Indentation level for pretty printing
    """
    logger.info(f"🔄 转换 JSONL 到 JSON: {jsonl_file} -> {json_file}")
    
    results = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                results.append(json.loads(line))
    
    # Save as standard JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"✅ 转换完成，共 {len(results)} 条记录")
    return len(results)


async def process_dataset(input_file: str, output_file: str, max_samples: int = None):
    """
    Process the dataset and generate submission file using JSON Lines format.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (will be converted from JSONL at the end)
        max_samples: Maximum number of samples to process (for testing)
    """
    logger.info(f"🚀 开始处理数据集: {input_file}")
    
    # Load dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        logger.info(f"📊 处理前 {max_samples} 个样本")
    
    # Initialize extractor
    extractor = HierarchicalEntityExtractor()
    
    # JSONL temporary file
    jsonl_file = output_file.replace('.json', '.jsonl')
    
    # Clear/create JSONL file (overwrite mode at start)
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        pass  # Create empty file
    
    total_samples = len(data)
    processed_count = 0
    
    # Show type mapping statistics for the first sample
    if data:
        first_sample = data[0]
        logger.info(f"\n📊 第一个样本的类型映射统计:")
        extractor.show_type_mapping_stats(first_sample['coarse_types'])
    
    for i, sample in enumerate(data, 1):
        # print(f"\n📝 处理样本 {i}/{total_samples}: {sample['id']}")
        
        sentence = sample['sentence']
        coarse_types = sample['coarse_types']
        
        # Show type mapping for this sample (only for first few samples)
        if i <= 3:
            extractor.show_type_mapping_stats(coarse_types)
        
        # Extract entities
        entities = await extractor.extract_entities_for_sentence(sentence, coarse_types)
        
        # Clean entities: remove debug fields like 'source'
        cleaned_entities = []
        for entity in entities:
            cleaned_entity = {
                'name': entity['name'],
                'coarse_type': entity['coarse_type'],
                'fine_type': entity['fine_type']
            }
            # Optionally include description if present
            if 'description' in entity and entity['description']:
                cleaned_entity['description'] = entity['description']
            cleaned_entities.append(cleaned_entity)
        
        # Create result entry with id
        # Preserve original id or generate new one
        sample_id = sample.get('id', str(uuid.uuid4()))
        
        result_entry = {
            "id": sample_id,
            "sentence": sentence,
            "entities": cleaned_entities
        }
        
        # Append to JSONL file (one line per result)
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
        
        processed_count += 1
        
        # Print progress
        if i % 10 == 0:
            logger.info(f"📈 已处理 {i}/{total_samples} 个样本")
    
    logger.info(f"\n✅ 处理完成! JSONL 文件: {jsonl_file}")
    logger.info(f"📊 总共处理了 {processed_count} 个样本")
    
    # Convert JSONL to standard JSON
    logger.info(f"\n🔄 转换为标准 JSON 格式...")
    total_records = jsonl_to_json(jsonl_file, output_file, indent=4)
    
    logger.info(f"✅ 最终结果保存到: {output_file}")
    
    # Calculate statistics from JSONL file
    total_entities = 0
    sample_results = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                result = json.loads(line)
                total_entities += len(result['entities'])
                if line_num <= 3:
                    sample_results.append(result)
    
    logger.info(f"📊 总共提取了 {total_entities} 个实体")
    
    # Show some examples
    logger.info(f"\n📋 示例结果:")
    for i, result in enumerate(sample_results):
        logger.info(f"  样本 {i+1}: {len(result['entities'])} 个实体")
        for entity in result['entities'][:2]:  # Show first 2 entities
            logger.info(f"    - {entity['name']}: {entity['coarse_type']} -> {entity['fine_type']}")


async def main():
    """Main function to run the extraction process."""
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    
    # For testing, process only first 10 samples
    # For full processing, set max_samples=None
    await process_dataset(input_file, output_file, max_samples=3)
    
    logger.info(f"\n🎉 完成! 检查结果文件: {output_file}")


async def process_full_dataset():
    """Process the entire dataset (use with caution - will take a long time)."""
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE
    
    print("⚠️  警告: 这将处理整个数据集，可能需要很长时间...")
    response = input("是否继续? (y/N): ")
    
    if response.lower() == 'y':
        await process_dataset(input_file, output_file, max_samples=None)
        logger.info(f"\n🎉 完整数据集处理完成! 检查结果文件: {output_file}")
    else:
        print("❌ 已取消处理")


if __name__ == "__main__":
    asyncio.run(main())
