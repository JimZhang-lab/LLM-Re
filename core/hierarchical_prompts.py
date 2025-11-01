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

# 英文第一阶段粗粒度抽取示例（覆盖主要类型）
HIERARCHICAL_PROMPTS["coarse_extraction_examples_en"] = [
    """<Example 1: science>
Text: "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4, addition of C4 - C - N - PEG9 into wormlike micelles reduces the critical packing parameter."
Available coarse types: biology, science, film information, restaurant information, product, other, 其他

<Output>
entity{tuple_delimiter}C4 - C - N - PEG9{tuple_delimiter}science{tuple_delimiter}C4 - C - N - PEG9 is a chemical compound with specific structural properties.
entity{tuple_delimiter}C12EO4{tuple_delimiter}science{tuple_delimiter}C12EO4 is a chemical compound used for comparison.
{completion_delimiter}
""",
    """<Example 2: location>
Text: "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available coarse types: literature, vehicle information, biology, film information, location, other, 其他

<Output>
entity{tuple_delimiter}China{tuple_delimiter}location{tuple_delimiter}China is a country where bai lan flowers are traditionally used.
{completion_delimiter}
""",
    """<Example 3: biology>
Text: "Rhododendron hippophaeoides (灰背杜鹃) is a species of flowering plant in the Ericaceae family."
Available coarse types: literature, medicine, biology, science, computer science, location, music, other, 其他

<Output>
entity{tuple_delimiter}flowering plant{tuple_delimiter}biology{tuple_delimiter}Flowering plant is a type of plant that produces flowers.
entity{tuple_delimiter}Ericaceae{tuple_delimiter}biology{tuple_delimiter}Ericaceae is a plant family.
{completion_delimiter}
""",
    """<Example 4: organization>
Text: "A version of HIIT was based on a 1996 study by Ritsumeikan University Professor Izumi Tabata."
Available coarse types: literature, organization, medicine, biology, science, food, politics, person, other, 其他

<Output>
entity{tuple_delimiter}Ritsumeikan University{tuple_delimiter}organization{tuple_delimiter}Ritsumeikan University is the institution where the study was conducted.
entity{tuple_delimiter}Izumi Tabata{tuple_delimiter}person{tuple_delimiter}Izumi Tabata is a professor who conducted the study.
{completion_delimiter}
""",
    """<Example 5: food>
Text: "The national dish of Bhutan, ema datshi, is made from homemade yak or mare milk cheese and hot peppers."
Available coarse types: food, politics, time, location, other, 其他

<Output>
entity{tuple_delimiter}ema datshi{tuple_delimiter}food{tuple_delimiter}Ema datshi is the national dish of Bhutan.
{completion_delimiter}
""",
    """<Example 6: person>
Text: "Nobel-laureate William Vickrey is considered by some to be the father of congestion pricing, as he first proposed adding tolls to the New York City Subway."
Available coarse types: person, organization, location, other, 其他

<Output>
entity{tuple_delimiter}William Vickrey{tuple_delimiter}person{tuple_delimiter}William Vickrey is a Nobel laureate known for congestion pricing theory.
entity{tuple_delimiter}New York City Subway{tuple_delimiter}location{tuple_delimiter}New York City Subway is a transportation facility.
{completion_delimiter}
""",
    """<Example 7: politics>
Text: "The Treasury's benchmark 30-year bond gained nearly half a point, or about $5 for each $1,000 face amount."
Available coarse types: finance, politics, organization, other, 其他

<Output>
entity{tuple_delimiter}Treasury{tuple_delimiter}politics{tuple_delimiter}Treasury is the government entity responsible for financial policy.
{completion_delimiter}
""",
    """<Example 8: event>
Text: "None were built prior to Japan's surrender and the end of World War II."
Available coarse types: event, time, location, other, 其他

<Output>
entity{tuple_delimiter}World War II{tuple_delimiter}event{tuple_delimiter}World War II was a major global conflict.
{completion_delimiter}
""",
]

# 中文第一阶段粗粒度抽取示例（覆盖主要类型）
HIERARCHICAL_PROMPTS["coarse_extraction_examples_zh"] = [
    """<示例 1: 位置+职位>
文本: "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
可用粗粒度类型: 生物, 职位, 科学, 组织机构, 学历, 位置, other, 其他

<输出>
entity{tuple_delimiter}深圳{tuple_delimiter}位置{tuple_delimiter}深圳是文中提到的城市。
entity{tuple_delimiter}记者{tuple_delimiter}职位{tuple_delimiter}记者是报道新闻的专业人员。
{completion_delimiter}
""",
    """<示例 2: 组织机构>
文本: "据我所知，宝钢对于衍生品市场的操作相对比较谨慎，应该不会在期货市场上有很大损失。"
可用粗粒度类型: 职位, 科学, 组织机构, 产品, 人, 事件, 位置, other, 其他

<输出>
entity{tuple_delimiter}宝钢{tuple_delimiter}组织机构{tuple_delimiter}宝钢是一家对衍生品市场操作谨慎的公司。
{completion_delimiter}
""",
    """<示例 3: 位置（多个国家）>
文本: "随后，日本、韩国、英国、德国等国际大公司纷纷前来，与他们签订合资开发细铅笔、纸杆铅笔、水溶铅笔等高档产品项目。"
可用粗粒度类型: 职位, 组织机构, 人, 文学, 事件, 位置, other, 其他

<输出>
entity{tuple_delimiter}日本{tuple_delimiter}位置{tuple_delimiter}日本是参与合作开发的国家之一。
entity{tuple_delimiter}韩国{tuple_delimiter}位置{tuple_delimiter}韩国是参与合作开发的国家之一。
entity{tuple_delimiter}英国{tuple_delimiter}位置{tuple_delimiter}英国是参与合作开发的国家之一。
entity{tuple_delimiter}德国{tuple_delimiter}位置{tuple_delimiter}德国是参与合作开发的国家之一。
{completion_delimiter}
""",
    """<示例 4: 人>
文本: "面对美国国会某些人的反华噪音，克林顿决定提前、专程对中国作长达8天的国事访问，这充分说明，他已认识到建立美中建设性战略伙伴关系的重要性。"
可用粗粒度类型: 职位, 组织机构, 人, 政治, 位置, other, 其他

<输出>
entity{tuple_delimiter}克林顿{tuple_delimiter}人{tuple_delimiter}克林顿是决定访华的政治家。
entity{tuple_delimiter}中国{tuple_delimiter}位置{tuple_delimiter}中国是访问的目的地国家。
entity{tuple_delimiter}美国{tuple_delimiter}位置{tuple_delimiter}美国是克林顿所代表的国家。
{completion_delimiter}
""",
    """<示例 5: 产品>
文本: "Q：现在war3的比赛越来越少，大赛中也仅仅留下了WCG、ESWC。"
可用粗粒度类型: 产品, 事件, 组织机构, other, 其他

<输出>
entity{tuple_delimiter}war3{tuple_delimiter}产品{tuple_delimiter}war3是一款电子竞技游戏。
entity{tuple_delimiter}WCG{tuple_delimiter}事件{tuple_delimiter}WCG是电子竞技赛事。
entity{tuple_delimiter}ESWC{tuple_delimiter}事件{tuple_delimiter}ESWC是电子竞技赛事。
{completion_delimiter}
""",
    """<示例 6: 文学>
文本: "不久前，围绕新拍《水浒》中潘金莲的形象设计争论不已。"
可用粗粒度类型: 文学, 产品, 人, other, 其他

<输出>
entity{tuple_delimiter}《水浒》{tuple_delimiter}文学{tuple_delimiter}《水浒》是中国古典名著。
entity{tuple_delimiter}潘金莲{tuple_delimiter}文学{tuple_delimiter}潘金莲是《水浒》中的角色。
{completion_delimiter}
""",
    """<示例 7: 事件>
文本: "最近三场比赛切尔西未尝胜果，下周又要打欧冠主场对克卢日的关键一战，球队急需一场胜利来鼓舞士气。"
可用粗粒度类型: 事件, 组织机构, 位置, other, 其他

<输出>
entity{tuple_delimiter}欧冠{tuple_delimiter}事件{tuple_delimiter}欧冠是欧洲足球冠军联赛。
entity{tuple_delimiter}切尔西{tuple_delimiter}组织机构{tuple_delimiter}切尔西是参赛的足球队。
{completion_delimiter}
""",
    """<示例 8: 生物>
文本: "5月1日凌晨，北京动物园9岁的东北虎"继生"怀孕105天后，顺利产下4只小虎仔。"
可用粗粒度类型: 生物, 位置, 组织机构, other, 其他

<输出>
entity{tuple_delimiter}继生{tuple_delimiter}生物{tuple_delimiter}继生是文中提到的东北虎。
entity{tuple_delimiter}北京动物园{tuple_delimiter}组织机构{tuple_delimiter}北京动物园是饲养该老虎的地点。
{completion_delimiter}
""",
]

# Stage 2: Fine Type Extraction (per coarse type)
# 单实体细粒度分类提示词（三元组格式）
HIERARCHICAL_PROMPTS["fine_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for classifying an entity into a fine-grained type.

---Instructions---
1.  **Entity Classification & Output:**
    *   **Identification:** Classify the entity "{entity_name}" (coarse type: {coarse_type}) into the most specific fine-grained type.
    *   **Entity Details:** Output:
        *   `entity_name`: The name of the entity (keep original).
        *   `fine_type`: Select the **most specific fine-grained type** from the available types.
        *   `entity_description`: A concise description based on the input text.
    *   **Output Format:** Output 4 fields delimited by `{tuple_delimiter}`:
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}fine_type{tuple_delimiter}entity_description`

2.  **Language & Proper Nouns:**
    *   Output language: `{language}`
    *   Keep proper nouns in their original language

3.  **Completion Signal:** Output `{completion_delimiter}` after classification.

---Examples---
{examples}

---Real Data---
<Input>
Text: {sentence}

Entity to classify: "{entity_name}"
Coarse type: {coarse_type}

Available fine-grained types for {coarse_type}: {fine_types}
```
"""

# 批量细粒度抽取提示词（优化版，使用三元组格式）
HIERARCHICAL_PROMPTS["fine_extraction_batch_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for classifying entities into fine-grained types.

---Instructions---
1.  **Entity Classification & Output:**
    *   **Identification:** For each entity provided, classify it into the most specific fine-grained type.
    *   **Entity Details:** For each entity, output:
        *   `entity_name`: The name of the entity (keep original).
        *   `fine_type`: Select the **most specific fine-grained type** from the available types for this entity's coarse type.
        *   `entity_description`: A concise description based on the input text.
    *   **Output Format:** Output 4 fields delimited by `{tuple_delimiter}`:
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}fine_type{tuple_delimiter}entity_description`

2.  **Language & Proper Nouns:**
    *   Output language: `{language}`
    *   Keep proper nouns in their original language

3.  **Completion Signal:** Output `{completion_delimiter}` after all entities have been classified.

---Examples---
{examples}

---Real Data---
<Input>
Text: {sentence}

Entities to classify (Coarse Type: {coarse_type}):
{entities_list}

Available fine-grained types for {coarse_type}: {fine_types}
```
"""

# 中文细粒度分类示例（从zh_data_train1.json选取，第二阶段单实体分类，三元组格式）
HIERARCHICAL_PROMPTS["fine_extraction_examples_zh"] = {
    "人": """<示例>
实体: "克林顿" (粗粒度: 人)
文本: "面对美国国会某些人的反华噪音，克林顿决定提前、专程对中国作长达8天的国事访问。"
可用细粒度类型: 专家, 学者, 政治家, 官员, 研究者, 科学家, 教授, 企业家

<输出>
entity{tuple_delimiter}克林顿{tuple_delimiter}政治家{tuple_delimiter}克林顿是决定访华的美国政治家。
{completion_delimiter}""",
    
    "职位": """<示例>
实体: "记者" (粗粒度: 职位)
文本: "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
可用细粒度类型: 概念

<输出>
entity{tuple_delimiter}记者{tuple_delimiter}概念{tuple_delimiter}记者是报道新闻的职位概念。
{completion_delimiter}""",
    
    "位置": """<示例>
实体: "日本" (粗粒度: 位置)
文本: "随后，日本、韩国、英国、德国等国际大公司纷纷前来。"
可用细粒度类型: 国家, 城市, 省份, 地区, 乡镇, 村庄, 州, 洲

<输出>
entity{tuple_delimiter}日本{tuple_delimiter}国家{tuple_delimiter}日本是参与合作的国家之一。
{completion_delimiter}""",
    
    "组织机构": """<示例>
实体: "宝钢" (粗粒度: 组织机构)
文本: "据我所知，宝钢对于衍生品市场的操作相对比较谨慎，应该不会在期货市场上有很大损失。"
可用细粒度类型: 政府, 大学, 公司, 银行, 企业, 组织, 机构, 军队, 媒体

<输出>
entity{tuple_delimiter}宝钢{tuple_delimiter}公司{tuple_delimiter}宝钢是一家对衍生品市场操作谨慎的公司。
{completion_delimiter}""",
    
    "科学": """<示例>
实体: "本科学历" (粗粒度: 科学)
文本: "2007年7月加入工作，本科学历，自动化仪表高级工程师，精通热能工程专业。"
可用细粒度类型: 学历, 专业, 学科, 概念, 奖项, 研究机构

<输出>
entity{tuple_delimiter}本科学历{tuple_delimiter}学历{tuple_delimiter}本科学历表示教育程度。
{completion_delimiter}""",
    
    "产品": """<示例>
实体: "war3" (粗粒度: 产品)
文本: "Q：现在war3的比赛越来越少，大赛中也仅仅留下了WCG、ESWC。"
可用细粒度类型: 软件, 电影, 军舰, 船, 航天器, 邮票

<输出>
entity{tuple_delimiter}war3{tuple_delimiter}软件{tuple_delimiter}war3是一款电子竞技游戏软件。
{completion_delimiter}""",
    
    "文学": """<示例>
实体: "潘金莲" (粗粒度: 文学)
文本: "不久前，围绕新拍《水浒》中潘金莲的形象设计争论不已。"
可用细粒度类型: 角色, 作家, 书, 画, 诗, 杂志, 漫画, 书法家, 画家

<输出>
entity{tuple_delimiter}潘金莲{tuple_delimiter}角色{tuple_delimiter}潘金莲是《水浒》中的角色。
{completion_delimiter}""",
    
    "事件": """<示例>
实体: "欧冠" (粗粒度: 事件)
文本: "最近三场比赛切尔西未尝胜果，下周又要打欧冠主场对克卢日的关键一战。"
可用细粒度类型: 活动

<输出>
entity{tuple_delimiter}欧冠{tuple_delimiter}活动{tuple_delimiter}欧冠是欧洲足球冠军联赛活动。
{completion_delimiter}""",
    
    "生物": """<示例>
实体: "继生" (粗粒度: 生物)
文本: "5月1日凌晨，北京动物园9岁的东北虎"继生"怀孕105天后，顺利产下4只小虎仔。"
可用细粒度类型: 动物

<输出>
entity{tuple_delimiter}继生{tuple_delimiter}动物{tuple_delimiter}继生是北京动物园的东北虎。
{completion_delimiter}""",
    
    "政治": """<示例>
实体: "政府" (粗粒度: 政治)
文本: "政府宣布新的经济政策将于下月实施。"
可用细粒度类型: 政府, 政治团体, 政治家, 官员, 法官, 革命家

<输出>
entity{tuple_delimiter}政府{tuple_delimiter}政府{tuple_delimiter}政府是宣布经济政策的机构。
{completion_delimiter}""",
}

# 英文细粒度分类示例（从en_data_train1.json选取，第二阶段单实体分类，三元组格式）
HIERARCHICAL_PROMPTS["fine_extraction_examples_en"] = {
    "location": """<Example>
Entity: "China" (coarse: location)
Text: "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available fine types: address, city, country, province, region, town, village, district, facility

<Output>
entity{tuple_delimiter}China{tuple_delimiter}country{tuple_delimiter}China is a country where bai lan flowers are traditionally used.
{completion_delimiter}""",
    
    "organization": """<Example>
Entity: "Ritsumeikan University" (coarse: organization)
Text: "A version of HIIT was based on a 1996 study by Ritsumeikan University Professor Izumi Tabata."
Available fine types: university, company, government, bank, school, academy, institution, media, team

<Output>
entity{tuple_delimiter}Ritsumeikan University{tuple_delimiter}university{tuple_delimiter}Ritsumeikan University is the institution where the study was conducted.
{completion_delimiter}""",
    
    "science": """<Example>
Entity: "C4 - C - N - PEG9" (coarse: science)
Text: "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4."
Available fine types: chemical, compound, element, formula, material, technology, concept, discipline

<Output>
entity{tuple_delimiter}C4 - C - N - PEG9{tuple_delimiter}chemical{tuple_delimiter}C4 - C - N - PEG9 is a chemical compound with specific structural properties.
{completion_delimiter}""",
    
    "biology": """<Example>
Entity: "flowering plant" (coarse: biology)
Text: "Rhododendron hippophaeoides is a species of flowering plant in the Ericaceae family."
Available fine types: DNA, RNA, animal, bacteria, plant, organism, species, gene, protein

<Output>
entity{tuple_delimiter}flowering plant{tuple_delimiter}plant{tuple_delimiter}Flowering plant is a type of plant that produces flowers.
{completion_delimiter}""",
    
    "person": """<Example>
Entity: "William Vickrey" (coarse: person)
Text: "Nobel-laureate William Vickrey is considered by some to be the father of congestion pricing."
Available fine types: researcher, expert, scholar, professor, scientist, engineer, economist, politician

<Output>
entity{tuple_delimiter}William Vickrey{tuple_delimiter}economist{tuple_delimiter}William Vickrey is a Nobel laureate economist known for congestion pricing theory.
{completion_delimiter}""",
    
    "food": """<Example>
Entity: "ema datshi" (coarse: food)
Text: "The national dish of Bhutan, ema datshi, is made from homemade yak or mare milk cheese and hot peppers."
Available fine types: dish, fruit, vegetable, beverage, dessert, condiment, dairy, meat

<Output>
entity{tuple_delimiter}ema datshi{tuple_delimiter}dish{tuple_delimiter}Ema datshi is the national dish of Bhutan.
{completion_delimiter}""",
    
    "event": """<Example>
Entity: "World War II" (coarse: event)
Text: "None were built prior to Japan's surrender and the end of World War II."
Available fine types: activity, war, conference, disaster, economy, politics, aid

<Output>
entity{tuple_delimiter}World War II{tuple_delimiter}war{tuple_delimiter}World War II was a major global conflict.
{completion_delimiter}""",
    
    "politics": """<Example>
Entity: "Treasury" (coarse: politics)
Text: "The Treasury's benchmark 30-year bond gained nearly half a point."
Available fine types: government, political party, politician, law, army, officer, judge

<Output>
entity{tuple_delimiter}Treasury{tuple_delimiter}government{tuple_delimiter}Treasury is the government entity responsible for financial policy.
{completion_delimiter}""",
    
    "product": """<Example>
Entity: "Ti-6Al-4V" (coarse: product)
Text: "Susceptibilities to delamination & warping of Ti-6Al-4V & Inconel 718 are examined."
Available fine types: material, vehicle, aircraft carrier, software, film, equipment unit, spacecraft

<Output>
entity{tuple_delimiter}Ti-6Al-4V{tuple_delimiter}material{tuple_delimiter}Ti-6Al-4V is a titanium alloy material.
{completion_delimiter}""",
    
    "medicine": """<Example>
Entity: "oligomenorrhoea" (coarse: medicine)
Text: "It is defined as the absence of menses for three months in a woman with previously normal menstruation."
Available fine types: disease, physician, apothecary

<Output>
entity{tuple_delimiter}oligomenorrhoea{tuple_delimiter}disease{tuple_delimiter}Oligomenorrhoea is a medical condition affecting menstruation.
{completion_delimiter}""",
    
    "time": """<Example>
Entity: "the past 15 years" (coarse: time)
Text: "During the past 15 years, it has gone from almost zilch to some 50% of production."
Available fine types: date, year, month, period, hours, specific time, concept

<Output>
entity{tuple_delimiter}the past 15 years{tuple_delimiter}period{tuple_delimiter}The past 15 years is a time period for measuring changes.
{completion_delimiter}""",
    
    "finance": """<Example>
Entity: "$ 100 million" (coarse: finance)
Text: "There are about $ 100 million of 7% term bonds due 2009."
Available fine types: money, concept, economist, entrepreneur, financial institution, investor

<Output>
entity{tuple_delimiter}$ 100 million{tuple_delimiter}money{tuple_delimiter}$ 100 million is a monetary amount.
{completion_delimiter}""",
    
    "music": """<Example>
Entity: "Vanessa L. Williams" (coarse: music)
Text: "The young artists spend a week learning from mentors like Vanessa L. Williams."
Available fine types: musician, band, song, violinist

<Output>
entity{tuple_delimiter}Vanessa L. Williams{tuple_delimiter}musician{tuple_delimiter}Vanessa L. Williams is a musician who mentors young artists.
{completion_delimiter}""",
    
    "literature": """<Example>
Entity: "Scylla" (coarse: literature)
Text: "Another mythological creature, the Scylla, is a similar female sea demon."
Available fine types: myth, character, book, poet, writer, painter, calligrapher, language

<Output>
entity{tuple_delimiter}Scylla{tuple_delimiter}myth{tuple_delimiter}Scylla is a mythological creature from ancient legends.
{completion_delimiter}""",
}

# 向后兼容：默认使用中文示例
HIERARCHICAL_PROMPTS["fine_extraction_examples"] = HIERARCHICAL_PROMPTS["fine_extraction_examples_zh"]

# 批量细粒度分类示例 - 中文（使用三元组格式，第二阶段批量分类）
HIERARCHICAL_PROMPTS["fine_extraction_batch_examples_zh"] = [
    """<示例 1: 职位>
文本: "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
待分类实体 (粗粒度: 职位): 记者
可用细粒度类型: 概念, 职业, 专家, 学者

<输出>
entity{tuple_delimiter}记者{tuple_delimiter}概念{tuple_delimiter}记者是报道新闻的职位概念。
{completion_delimiter}
""",
    """<示例 2: 位置>
文本: "随后，日本、韩国、英国、德国等国际大公司纷纷前来，与他们签订合资开发细铅笔、纸杆铅笔、水溶铅笔等高档产品项目。"
待分类实体 (粗粒度: 位置): 日本, 韩国, 英国, 德国
可用细粒度类型: 国家, 城市, 省份, 地区, 乡镇, 村庄

<输出>
entity{tuple_delimiter}日本{tuple_delimiter}国家{tuple_delimiter}日本是参与合作的国家。
entity{tuple_delimiter}韩国{tuple_delimiter}国家{tuple_delimiter}韩国是参与合作的国家。
entity{tuple_delimiter}英国{tuple_delimiter}国家{tuple_delimiter}英国是参与合作的国家。
entity{tuple_delimiter}德国{tuple_delimiter}国家{tuple_delimiter}德国是参与合作的国家。
{completion_delimiter}
""",
    """<示例 3: 科学>
文本: "2007年7月加入工作，本科学历，自动化仪表高级工程师，精通热能工程专业。"
待分类实体 (粗粒度: 科学): 本科学历, 热能工程专业
可用细粒度类型: 学历, 专业, 学科, 概念

<输出>
entity{tuple_delimiter}本科学历{tuple_delimiter}学历{tuple_delimiter}本科学历表示教育程度。
entity{tuple_delimiter}热能工程专业{tuple_delimiter}专业{tuple_delimiter}热能工程专业是工程领域的专业。
{completion_delimiter}
""",
    """<示例 4: 组织机构>
文本: "1963年11月出生，汉族，中共党员，大学专科学历，会计师。"
待分类实体 (粗粒度: 组织机构): 汉族
可用细粒度类型: 民族, 公司, 政府, 大学, 学校, 军队, 媒体, 团队

<输出>
entity{tuple_delimiter}汉族{tuple_delimiter}民族{tuple_delimiter}汉族是中国的主要民族之一。
{completion_delimiter}
""",
    """<示例 5: 位置（河流）>
文本: "总体上看，长江、珠江和黄河干流水质尚可，淮河干流和松花江水质有所好转。"
待分类实体 (粗粒度: 位置): 长江, 珠江, 黄河, 淮河, 松花江
可用细粒度类型: 河, 湖, 海, 山, 城市, 国家, 地区

<输出>
entity{tuple_delimiter}长江{tuple_delimiter}河{tuple_delimiter}长江是中国最长的河流。
entity{tuple_delimiter}珠江{tuple_delimiter}河{tuple_delimiter}珠江是华南地区的重要河流。
entity{tuple_delimiter}黄河{tuple_delimiter}河{tuple_delimiter}黄河是中国第二长河流。
entity{tuple_delimiter}淮河{tuple_delimiter}河{tuple_delimiter}淮河是中国重要河流之一。
entity{tuple_delimiter}松花江{tuple_delimiter}河{tuple_delimiter}松花江是东北地区的主要河流。
{completion_delimiter}
""",
]

# 批量细粒度分类示例 - 英文（使用三元组格式，第二阶段批量分类）
HIERARCHICAL_PROMPTS["fine_extraction_batch_examples_en"] = [
    """<Example 1: location>
Text: "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Entities to classify (Coarse Type: location): China
Available fine types: country, city, province, region, town, village, district

<Output>
entity{tuple_delimiter}China{tuple_delimiter}country{tuple_delimiter}China is a country where bai lan flowers are used.
{completion_delimiter}
""",
    """<Example 2: science>
Text: "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4."
Entities to classify (Coarse Type: science): C4 - C - N - PEG9, C12EO4
Available fine types: chemical, compound, element, formula, material

<Output>
entity{tuple_delimiter}C4 - C - N - PEG9{tuple_delimiter}chemical{tuple_delimiter}C4 - C - N - PEG9 is a chemical compound with specific properties.
entity{tuple_delimiter}C12EO4{tuple_delimiter}chemical{tuple_delimiter}C12EO4 is a chemical compound used for comparison.
{completion_delimiter}
""",
    """<Example 3: biology>
Text: "Rhododendron hippophaeoides (灰背杜鹃) is a species of flowering plant in the Ericaceae family."
Entities to classify (Coarse Type: biology): flowering plant, Ericaceae
Available fine types: plant, animal, bacteria, organism, species

<Output>
entity{tuple_delimiter}flowering plant{tuple_delimiter}plant{tuple_delimiter}Flowering plant is a type of plant that produces flowers.
entity{tuple_delimiter}Ericaceae{tuple_delimiter}plant{tuple_delimiter}Ericaceae is a plant family.
{completion_delimiter}
""",
    """<Example 4: person>
Text: "The young artists spend a week learning from mentors like Mikhail Baryshnikov, Vanessa L. Williams, Jacques d'Amboise, and Michael Tilson Thomas."
Entities to classify (Coarse Type: person): Mikhail Baryshnikov, Jacques d'Amboise
Available fine types: dancer, musician, actor, artist, athlete, coach, conductor

<Output>
entity{tuple_delimiter}Mikhail Baryshnikov{tuple_delimiter}dancer{tuple_delimiter}Mikhail Baryshnikov is a renowned dancer and mentor.
entity{tuple_delimiter}Jacques d'Amboise{tuple_delimiter}dancer{tuple_delimiter}Jacques d'Amboise is a professional dancer.
{completion_delimiter}
""",
    """<Example 5: product>
Text: "Susceptibilities to delamination & warping of Ti-6Al-4V & Inconel 718 are examined."
Entities to classify (Coarse Type: product): Ti-6Al-4V, Inconel 718
Available fine types: material, vehicle, equipment unit, software, film, spacecraft

<Output>
entity{tuple_delimiter}Ti-6Al-4V{tuple_delimiter}material{tuple_delimiter}Ti-6Al-4V is a titanium alloy material.
entity{tuple_delimiter}Inconel 718{tuple_delimiter}material{tuple_delimiter}Inconel 718 is a nickel-based superalloy material.
{completion_delimiter}
""",
]

# ============== Reverse Extraction Prompts (Fine -> Coarse) ==============

# 反向抽取系统提示词：直接抽取细粒度类型实体
HIERARCHICAL_PROMPTS["reverse_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities with fine-grained type classification.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify all clearly defined entities in the input text.
    *   **Entity Details:** For each identified entity, extract:
        *   `entity_name`: The name of the entity (use title case for proper nouns).
        *   `entity_type`: Select the **most specific fine-grained type** from: `{entity_types}`.
        *   `entity_description`: A concise description based on the input text.
    *   **Type Selection:** Choose the most appropriate fine-grained type that specifically describes the entity.
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
Available_fine_types: [{entity_types}]
Text:
```
{input_text}
```
"""

# 中文反向抽取示例（从zh_data_train1.json选取，直接抽取细粒度类型）
HIERARCHICAL_PROMPTS["reverse_extraction_examples_zh"] = [
    """<示例 1>
文本: "截至9月末，深圳现金累计投放量同比出现负数。近日，一位接近监管部门人士对本报记者称。"
可用细粒度类型: 概念, 职业, 专家, 学者, 记者, 编辑, 作家, 城市, 省份, 地区, 国家

<输出>
entity{tuple_delimiter}深圳{tuple_delimiter}城市{tuple_delimiter}深圳是文中提到的城市。
entity{tuple_delimiter}记者{tuple_delimiter}概念{tuple_delimiter}记者是报道新闻的职位概念。
{completion_delimiter}
""",
    """<示例 2>
文本: "据我所知，宝钢对于衍生品市场的操作相对比较谨慎，应该不会在期货市场上有很大损失。"
可用细粒度类型: 公司, 企业, 银行, 政府, 大学, 组织, 机构, 概念

<输出>
entity{tuple_delimiter}宝钢{tuple_delimiter}公司{tuple_delimiter}宝钢是一家对衍生品市场操作谨慎的公司。
{completion_delimiter}
""",
    """<示例 3>
文本: "2007年7月加入工作，本科学历，自动化仪表高级工程师，精通热能工程专业。"
可用细粒度类型: 学历, 专业, 学科, 概念, 职业

<输出>
entity{tuple_delimiter}本科学历{tuple_delimiter}学历{tuple_delimiter}本科学历表示教育程度。
entity{tuple_delimiter}自动化仪表高级工程师{tuple_delimiter}专业{tuple_delimiter}自动化仪表高级工程师是一个专业职位。
entity{tuple_delimiter}热能工程专业{tuple_delimiter}专业{tuple_delimiter}热能工程专业是一个工程领域。
{completion_delimiter}
""",
    """<示例 4>
文本: "随后，日本、韩国、英国、德国等国际大公司纷纷前来，与他们签订合资开发项目。"
可用细粒度类型: 国家, 城市, 省份, 地区, 乡镇, 村庄, 大陆, 区域

<输出>
entity{tuple_delimiter}日本{tuple_delimiter}国家{tuple_delimiter}日本是参与合作的国家。
entity{tuple_delimiter}韩国{tuple_delimiter}国家{tuple_delimiter}韩国是参与合作的国家。
entity{tuple_delimiter}英国{tuple_delimiter}国家{tuple_delimiter}英国是参与合作的国家。
entity{tuple_delimiter}德国{tuple_delimiter}国家{tuple_delimiter}德国是参与合作的国家。
{completion_delimiter}
""",
    """<示例 5>
文本: "曾任矿冶总院自动化室工程师、人事处处长、党委副书记。"
可用细粒度类型: 概念, 职业, 官员, 专家, 研究者, 工程师, 机构, 组织

<输出>
entity{tuple_delimiter}矿冶总院{tuple_delimiter}机构{tuple_delimiter}矿冶总院是研究机构。
entity{tuple_delimiter}自动化室工程师{tuple_delimiter}概念{tuple_delimiter}自动化室工程师是工程技术岗位概念。
entity{tuple_delimiter}人事处处长{tuple_delimiter}概念{tuple_delimiter}人事处处长是管理岗位概念。
entity{tuple_delimiter}党委副书记{tuple_delimiter}概念{tuple_delimiter}党委副书记是党内职务概念。
{completion_delimiter}
""",
]

# 英文反向抽取示例（从en_data_train1.json选取，直接抽取细粒度类型）
HIERARCHICAL_PROMPTS["reverse_extraction_examples_en"] = [
    """<Example 1>
Text: "Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4, addition of C4 - C - N - PEG9 into wormlike micelles reduces the critical packing parameter."
Available fine types: chemical, compound, element, molecule, substance, material, equipment, technology

<Output>
entity{tuple_delimiter}C4 - C - N - PEG9{tuple_delimiter}chemical{tuple_delimiter}C4 - C - N - PEG9 is a chemical compound with specific structural properties.
entity{tuple_delimiter}C12EO4{tuple_delimiter}chemical{tuple_delimiter}C12EO4 is a chemical compound used for comparison.
{completion_delimiter}
""",
    """<Example 2>
Text: "In China, where it is known as bai lan (白蘭), the flowers are used to prepare yulan tea."
Available fine types: country, city, province, region, town, village, district, area

<Output>
entity{tuple_delimiter}China{tuple_delimiter}country{tuple_delimiter}China is a country where bai lan flowers are traditionally used.
{completion_delimiter}
""",
    """<Example 3>
Text: "Rhododendron hippophaeoides (灰背杜鹃) is a species of flowering plant in the Ericaceae family."
Available fine types: plant, animal, bacteria, virus, organism, species, genus, family

<Output>
entity{tuple_delimiter}flowering plant{tuple_delimiter}plant{tuple_delimiter}Flowering plant is a type of plant that produces flowers.
entity{tuple_delimiter}Ericaceae{tuple_delimiter}plant{tuple_delimiter}Ericaceae is a plant family.
{completion_delimiter}
""",
    """<Example 4>
Text: "A version of HIIT was based on a 1996 study by Ritsumeikan University Professor Izumi Tabata."
Available fine types: university, company, government, bank, school, academy, institution, organization

<Output>
entity{tuple_delimiter}Ritsumeikan University{tuple_delimiter}university{tuple_delimiter}Ritsumeikan University is the institution where the study was conducted.
{completion_delimiter}
""",
    """<Example 5>
Text: "The pineapple bun may be pre-stuffed with red bean paste, custard cream, or a sweet filling like that in a cocktail bun."
Available fine types: snack, dish, drink, ingredient, meal, dessert

<Output>
entity{tuple_delimiter}pineapple bun{tuple_delimiter}snack{tuple_delimiter}Pineapple bun is a type of bread or pastry.
entity{tuple_delimiter}cocktail bun{tuple_delimiter}snack{tuple_delimiter}Cocktail bun is a type of sweet pastry.
{completion_delimiter}
""",
]

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

# Continue extraction prompt (for gleaning) - 粗粒度
HIERARCHICAL_PROMPTS["coarse_continue_extraction_prompt"] = """---Task---
You have already extracted the following entities from the text:

{extracted_entities_list}

Now, carefully review the text again and extract ANY MISSED entities that were NOT in the above list.

---Text---
{sentence}

---Available Coarse Types---
{coarse_types}

---Instructions---
1. ONLY output entities that are NOT in the already extracted list above
2. Use the same format: entity{tuple_delimiter}name{tuple_delimiter}coarse_type{tuple_delimiter}description
3. If there are NO missed entities, just output: {completion_delimiter}
4. Do NOT repeat any entity from the already extracted list

<Output>
"""

# Continue extraction prompt - 细粒度
HIERARCHICAL_PROMPTS["fine_continue_extraction_prompt"] = """---Task---
You have already extracted the following {coarse_type} entities from the text:

{extracted_entities_list}

Now, carefully review the text again and extract ANY MISSED {coarse_type} entities that were NOT in the above list.

---Text---
{sentence}

---Available Fine Types for {coarse_type}---
{fine_types}

---Instructions---
1. ONLY output {coarse_type} entities that are NOT in the already extracted list above
2. Output ONLY the entity names, one per line
3. If there are NO missed entities, just output: NONE
4. Do NOT repeat any entity from the already extracted list

<Output>
"""

# 旧版继续抽取提示词（向后兼容）
HIERARCHICAL_PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Extract any MISSED entities from the previous extraction.

---Instructions---
1. Only output entities that were NOT in the previous extraction
2. Use the most specific type available
3. Output format: entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description
4. End with `{completion_delimiter}`

<Output>
"""

