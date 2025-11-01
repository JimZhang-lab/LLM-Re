"""
Optimized prompts for hierarchical entity extraction with coarse-fine types.
Based on the training dataset format.
"""

HIERARCHICAL_PROMPTS = {}

# ============== Two-Stage Extraction Prompts ==============

# Stage 1: Coarse Type Extraction
HIERARCHICAL_PROMPTS["coarse_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and classifying them into coarse-grained types.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify all clearly defined entities in the input text.
    *   **Entity Details:** For each identified entity, extract:
        *   `entity_name`: The name of the entity (use title case for proper nouns).
        *   `entity_type`: Select the **coarse-grained type** from: `{entity_types}`.
        *   `entity_description`: A concise description based on the input text.
    *   **Type Selection:** Choose the most appropriate coarse type that broadly describes the entity.
    *   **Output Format:** Output 4 fields delimited by `{tuple_delimiter}`:
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Language & Proper Nouns:**
    *   Output language: `{language}`
    *   Keep proper nouns in their original language

3.  **Completion Signal:** Output `{completion_delimiter}` after all entities have been extracted.

---Examples---
{examples}

---Real Data---
<Input>
Available_coarse_types: [{entity_types}]
Text:
```
{input_text}
```
"""

HIERARCHICAL_PROMPTS["coarse_extraction_examples_en"] = [
    """<Example 1>
Text: "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4, addition of C4 - C - N - PEG9 into wormlike micelles reduces the critical packing parameter."
Available coarse types: biology, science, film information, restaurant information, product, other, 其他

<Output>
entity{tuple_delimiter}C4 - C - N - PEG9{tuple_delimiter}science{tuple_delimiter}C4 - C - N - PEG9 is a chemical compound with specific structural properties.
entity{tuple_delimiter}C12EO4{tuple_delimiter}science{tuple_delimiter}C12EO4 is a chemical compound used for comparison.
{completion_delimiter}
""",
    """<Example 2>
Text: "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available coarse types: literature, vehicle information, biology, film information, location, other, 其他

<Output>
entity{tuple_delimiter}China{tuple_delimiter}location{tuple_delimiter}China is a country where bai lan flowers are traditionally used.
{completion_delimiter}
""",
    """<Example 3>
Text: "Rhododendron hippophaeoides (灰背杜鹃) is a species of flowering plant in the Ericaceae family."
Available coarse types: literature, medicine, biology, science, computer science, location, music, other, 其他

<Output>
entity{tuple_delimiter}flowering plant{tuple_delimiter}biology{tuple_delimiter}Flowering plant is a type of plant that produces flowers.
entity{tuple_delimiter}Ericaceae{tuple_delimiter}biology{tuple_delimiter}Ericaceae is a plant family.
{completion_delimiter}
""",
]

HIERARCHICAL_PROMPTS["coarse_extraction_examples_zh"] = [
    """<示例 1>
文本: "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
可用粗粒度类型: 生物, 职位, 科学, 组织机构, 学历, 位置, other, 其他

<输出>
entity{tuple_delimiter}记者{tuple_delimiter}职位{tuple_delimiter}记者是报道新闻的专业人员。
{completion_delimiter}
""",
    """<示例 2>
文本: "据我所知，宝钢对于衍生品市场的操作相对比较谨慎，应该不会在期货市场上有很大损失。"
可用粗粒度类型: 职位, 科学, 组织机构, 产品, 人, 事件, 位置, other, 其他

<输出>
entity{tuple_delimiter}宝钢{tuple_delimiter}组织机构{tuple_delimiter}宝钢是一家对衍生品市场操作谨慎的公司。
{completion_delimiter}
""",
    """<示例 3>
文本: "随后，日本、韩国、英国、德国等国际大公司纷纷前来，与他们签订合资开发细铅笔、纸杆铅笔、水溶铅笔等高档产品项目。"
可用粗粒度类型: 职位, 组织机构, 人, 文学, 事件, 位置, other, 其他

<输出>
entity{tuple_delimiter}日本{tuple_delimiter}位置{tuple_delimiter}日本是参与合作开发的国家之一。
entity{tuple_delimiter}韩国{tuple_delimiter}位置{tuple_delimiter}韩国是参与合作开发的国家之一。
entity{tuple_delimiter}英国{tuple_delimiter}位置{tuple_delimiter}英国是参与合作开发的国家之一。
entity{tuple_delimiter}德国{tuple_delimiter}位置{tuple_delimiter}德国是参与合作开发的国家之一。
{completion_delimiter}
""",
]

# Stage 2: Fine Type Extraction (per coarse type)
HIERARCHICAL_PROMPTS["fine_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist performing fine-grained entity type classification.

---Task---
Classify the entity "{entity_name}" (which has coarse type: {coarse_type}) into the **most specific fine-grained type**.

---Context---
Text: {sentence}

Entity to classify: "{entity_name}"
Coarse type: {coarse_type}

Available fine-grained types for {coarse_type}: {fine_types}

---Instructions---
1. Analyze the context carefully to understand what "{entity_name}" represents
2. Select the MOST SPECIFIC fine-grained type that accurately describes this entity
3. Consider the semantic meaning and role of the entity in the context
4. Output ONLY the exact type name from the available fine types

---Output Format---
Type: [exact_fine_type_name]

---Examples---
{examples}

---Your Answer---
Type:"""

HIERARCHICAL_PROMPTS["fine_extraction_examples"] = {
    "person": """Example: Entity "薛瑞勇" (coarse: 人) in "薛瑞勇，男，汉族，1963年10月生，河北唐山人。"
Available fine types: 研究者, 专家, 学者, 教授, 官员, 企业家, 作家, 编辑, 记者, 运动员, 艺术家, 演员, 导演, 歌手, 音乐家, 画家, 科学家, 工程师, 医生, 律师
Answer: 研究者""",
    
    "location": """Example: Entity "China" (coarse: location) in "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available fine types: address, amenity, area, astronomical object, atoll, basin, bridge, canyon, castle, city, coast, community, company, concept, continent, country, crater, delta, desert, district, facility
Answer: country""",
    
    "organization": """Example: Entity "宝钢" (coarse: 组织机构) in "据我所知，宝钢对于衍生品市场的操作相对比较谨慎，应该不会在期货市场上有很大损失。"
Available fine types: 政府, 大学, 公司, 银行, 企业, 组织, 机构, 协会, 联盟, 团队, 媒体, 学校, 医院, 研究所, 实验室
Answer: 公司""",
    
    "science": """Example: Entity "C4 - C - N - PEG9" (coarse: science) in "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4."
Available fine types: algorithm, chemical, concept, element, equipment, field, formula, manufacturing process, material, method, metric, mineral, model, molecule, organic compound, physical quantity, principle, protein, research, research area, scientific law, structure, substance, system, technique, technology, theory, unit
Answer: chemical""",
    
    "biology": """Example: Entity "flowering plant" (coarse: biology) in "Rhododendron hippophaeoides is a species of flowering plant in the Ericaceae family."
Available fine types: DNA, RNA, anatomy, animal, bacteria, biologist, biomedical, bone, botanist, cell line, chemical, disease, ecology, enzyme, gene, geneticist, metabolite, microorganism, molecule, nucleotide, organ, organism, plant, process, protein, researcher, species, tissue, virus
Answer: plant""",
    
    "职位": """Example: Entity "记者" (coarse: 职位) in "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
Available fine types: 概念
Answer: 概念""",
}

# ============== Original Prompts (for backward compatibility) ==============

# Optimized entity extraction prompt for hierarchical types
HIERARCHICAL_PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities with hierarchical type classification (coarse and fine types).

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined entities in the input text.
    *   **Entity Details:** For each identified entity, extract:
        *   `entity_name`: The name of the entity (use title case for proper nouns).
        *   `entity_type`: Use the **most specific fine-grained type** from the available types: `{entity_types}`.
        *   `entity_description`: A concise description based on the input text.
    *   **Type Selection Priority:**
        *   ALWAYS select the most specific (fine-grained) type available
        *   If multiple fine types apply, choose the most relevant one based on context
        *   Only use coarse types if no specific fine type is available
    *   **Output Format:** Output 4 fields delimited by `{tuple_delimiter}`:
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Language & Proper Nouns:**
    *   Output language: `{language}`
    *   Keep proper nouns in their original language

3.  **Completion Signal:** Output `{completion_delimiter}` after all entities have been extracted.

---Examples---
{examples}

---Real Data---
<Input>
Available_types: [{entity_types}]
Text:
```
{input_text}
```
"""

# Optimized examples based on training data
HIERARCHICAL_PROMPTS["entity_extraction_examples_en"] = [
    """<Example 1>
Text: "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4, addition of C4 - C - N - PEG9 into wormlike micelles reduces the critical packing parameter."
Available types: chemical, compound, element, molecule, substance, material, equipment, technology

<Output>
entity{tuple_delimiter}C4 - C - N - PEG9{tuple_delimiter}chemical{tuple_delimiter}C4 - C - N - PEG9 is a chemical compound with specific structural properties.
entity{tuple_delimiter}C12EO4{tuple_delimiter}chemical{tuple_delimiter}C12EO4 is a chemical compound used for comparison.
{completion_delimiter}
""",
    """<Example 2>
Text: "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available types: country, city, province, region, town, village, district

<Output>
entity{tuple_delimiter}China{tuple_delimiter}country{tuple_delimiter}China is a country where bai lan flowers are traditionally used.
{completion_delimiter}
""",
    """<Example 3>
Text: "Rhododendron hippophaeoides (灰背杜鹃) is a species of flowering plant in the Ericaceae family."
Available types: plant, animal, bacteria, virus, organism, species, genus, family

<Output>
entity{tuple_delimiter}flowering plant{tuple_delimiter}plant{tuple_delimiter}Flowering plant is a type of plant that produces flowers.
entity{tuple_delimiter}Ericaceae{tuple_delimiter}plant{tuple_delimiter}Ericaceae is a plant family.
{completion_delimiter}
""",
]

HIERARCHICAL_PROMPTS["entity_extraction_examples_zh"] = [
    """<示例 1>
文本: "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
可用类型: 概念, 职业, 专家, 学者, 记者, 编辑, 作家

<输出>
entity{tuple_delimiter}记者{tuple_delimiter}概念{tuple_delimiter}记者是报道新闻的职位概念。
{completion_delimiter}
""",
    """<示例 2>
文本: "据我所知，宝钢对于衍生品市场的操作相对比较谨慎，应该不会在期货市场上有很大损失。"
可用类型: 公司, 企业, 银行, 政府, 大学, 组织, 机构

<输出>
entity{tuple_delimiter}宝钢{tuple_delimiter}公司{tuple_delimiter}宝钢是一家对衍生品市场操作谨慎的公司。
{completion_delimiter}
""",
    """<示例 3>
文本: "随后，日本、韩国、英国、德国等国际大公司纷纷前来，与他们签订合资开发细铅笔、纸杆铅笔、水溶铅笔等高档产品项目。"
可用类型: 国家, 城市, 省份, 地区, 村庄, 乡镇, 县城

<输出>
entity{tuple_delimiter}日本{tuple_delimiter}国家{tuple_delimiter}日本是参与合作开发的国家之一。
entity{tuple_delimiter}韩国{tuple_delimiter}国家{tuple_delimiter}韩国是参与合作开发的国家之一。
entity{tuple_delimiter}英国{tuple_delimiter}国家{tuple_delimiter}英国是参与合作开发的国家之一。
entity{tuple_delimiter}德国{tuple_delimiter}国家{tuple_delimiter}德国是参与合作开发的国家之一。
{completion_delimiter}
""",
]

# Re-extraction prompt for same coarse-fine types
HIERARCHICAL_PROMPTS["re_extraction_prompt"] = """---Role---
You are a Knowledge Graph Specialist performing fine-grained entity type classification.

---Task---
Classify the entity "{entity_name}" into the **most specific and appropriate** fine-grained type from the available options.

---Context---
Text: {sentence}

Entity to classify: "{entity_name}"

Available fine-grained types: {fine_types}

---Instructions---
1. Analyze the context carefully to understand what "{entity_name}" represents
2. Select the MOST SPECIFIC fine-grained type that accurately describes this entity
3. Consider the semantic meaning and role of the entity in the context
4. Output ONLY the exact type name from the available types, nothing else

---Output Format---
Type: [exact_type_name]

---Examples---
{examples}

---Your Answer---
Type:"""

HIERARCHICAL_PROMPTS["re_extraction_examples"] = {
    "person": """Example 1: Entity "薛瑞勇" in text "薛瑞勇，男，汉族，1963年10月生，河北唐山人。"
Available types: 研究者, 专家, 学者, 教授, 官员, 企业家
Answer: 研究者

Example 2: Entity "记者" in text "近日，一位接近监管部门人士对本报记者称。"
Available types: 概念, 职业, 专家, 学者
Answer: 概念""",
    
    "location": """Example 1: Entity "China" in text "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available types: city, town, village, country, province, region
Answer: country

Example 2: Entity "日本" in text "随后，日本、韩国、英国、德国等国际大公司纷纷前来。"
Available types: 国家, 城市, 省份, 地区, 大陆
Answer: 国家""",
    
    "organization": """Example 1: Entity "宝钢" in text "据我所知，宝钢对于衍生品市场的操作相对比较谨慎。"
Available types: 公司, 政府, 大学, 银行, 企业, 组织
Answer: 公司

Example 2: Entity "Ritsumeikan University" in text "A version of HIIT was based on a 1996 study by Ritsumeikan University Professor Izumi Tabata."
Available types: university, company, government, bank, school, academy
Answer: university""",
    
    "science": """Example 1: Entity "C4 - C - N - PEG9" in text "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4."
Available types: chemical, compound, element, molecule, substance, material
Answer: chemical

Example 2: Entity "C12EO4" in text "C12EO4 is used as a comparison compound in the study."
Available types: chemical, compound, element, molecule, substance
Answer: chemical""",
    
    "biology": """Example 1: Entity "flowering plant" in text "Rhododendron hippophaeoides is a species of flowering plant in the Ericaceae family."
Available types: plant, animal, bacteria, virus, organism, species
Answer: plant

Example 2: Entity "Ericaceae" in text "Rhododendron hippophaeoides is a species of flowering plant in the Ericaceae family."
Available types: plant, animal, bacteria, family, genus, species
Answer: plant""",
}

# User prompt for initial extraction
HIERARCHICAL_PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities from the input text.

---Output Requirements---
1. Output ONLY the extracted entities list
2. Use the most specific type available for each entity
3. End with `{completion_delimiter}`
4. No explanations or additional text

<Output>
"""

# Continue extraction prompt (for gleaning)
HIERARCHICAL_PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Extract any MISSED entities from the previous extraction.

---Instructions---
1. Only output entities that were NOT in the previous extraction
2. Use the most specific type available
3. Output format: entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description
4. End with `{completion_delimiter}`

<Output>
"""

