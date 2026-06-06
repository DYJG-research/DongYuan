import pandas as pd
import numpy as np
from openai import OpenAI
import json
import pandas as pd
import json
from openai import OpenAI
import time
import re
import os
from typing import Any, Dict
import json_repair
import re
from typing import Any, Dict
import json_repair
import random
# from embedding_utils import get_embedding
from multiprocessing import Pool, Manager
from tqdm import tqdm

def clean_response_to_json(text: str) -> Any:
    """
    清洗 LLM 返回文本并解析为 JSON (增强版)。
    能够自动修复未转义的引号、换行符以及 Markdown 标记干扰。
    支持解析单个对象 {} 或 列表 []。
    """
    if not text:
        return {}

    # 1. 移除 <think>...</think> 思考过程 (针对 DeepSeek/Qwen 等推理模型)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. 移除 Markdown 代码块标记 (```json ... ```)
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # 3. 提取最外层的 JSON 对象或数组
    # 这一步能过滤掉 LLM 在 JSON 前后的闲聊
    
    # 寻找可能的起始位置
    idx_obj_start = text.find('{')
    idx_arr_start = text.find('[')
    
    start_idx = -1
    end_idx = -1
    
    # 确定是对象还是数组在前
    if idx_obj_start != -1 and (idx_arr_start == -1 or idx_obj_start < idx_arr_start):
        start_idx = idx_obj_start
        end_idx = text.rfind('}')
    elif idx_arr_start != -1 and (idx_obj_start == -1 or idx_arr_start < idx_obj_start):
        start_idx = idx_arr_start
        end_idx = text.rfind(']')

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx : end_idx + 1]
    else:
        # 如果找不到花括号或方括号，可能格式彻底坏了，但还是试着扔给 repair 处理一下
        pass

    # 4. 使用 json_repair 进行容错解析
    # 它可以处理:
    # - 键值对里的未转义引号: "msg": "He said "Hello"" -> "msg": "He said \"Hello\""
    # - 字符串里的换行符
    # - 尾部多余的逗号
    try:
        return json_repair.loads(text)
    except Exception as e:
        # 如果 json_repair 都救不回来，那就真的没救了
        print(f"❌ JSON Repair failed. Raw text snippet: {text[:100]}...")
        raise e
import os

# ============================================================
# LLM API Configuration (override via environment variables)
# ============================================================
#   export LLM_API_BASE_URL=http://127.0.0.1:8000/v1
#   export LLM_MODEL_NAME=Qwen3-32B
#   export LLM_API_KEY=your_key_here
# ============================================================
BASE_URL = os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.getenv("LLM_MODEL_NAME", "Qwen3-32B")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
client = OpenAI(api_key=LLM_API_KEY, base_url=BASE_URL)

def llm(prompt, system="You are a helpful assistant."):
    global client
    if client is None:
        client = OpenAI(api_key=LLM_API_KEY, base_url=BASE_URL)
    messages=[]
    messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.5,
    max_tokens=10240,
    stream=False,
    # 不再传 tools，避免误触发工具导致响应结构变化
    ) 
    return response.choices[0].message.content
def  llm_generation_and_check(prompt,type=str, system="You are a helpful assistant."):
    resp_str = None
    retry_times = 0
    max_retries = 5

    while not isinstance(resp_str, type) and retry_times < max_retries:
        try:
            resp_str = llm(prompt,system)
            
        except Exception as e:
            retry_times += 1
            print(f"llm_judge generation error.Retry {retry_times}/{max_retries} times, due to error: {e}")

    # 如果重试多次仍然失败，返回空字符串，避免后续 split 报错
    if not isinstance(resp_str, type):
        return ""
    return resp_str

prompt_v1="""/no_think
假设你是一名中西医结合诊断脾胃病专家，擅长在问诊中分析患者提供的症状并对其进行分析和整理。请你参考【历史问诊信息】内容进行诊断分析，分析内容需遵循以下【思考与分析要求】。
【历史问诊信息】：{last_round_diagnosis}
【下一轮问诊问题】：{query}
【思考与分析要求】：
请你严格按照以下步骤和逻辑展开分析，确保逻辑清晰、因果明确：
1. **症状整体描述与分析**
    - 首先，完整分别描述患者已存在的所有症状(如有)和不存在的所有症状(没有不存在的症状就不要加后半句)，若患者有新补充的症状或否定的症状，需明确患者最新补充或否定的内容。
    - 随后，从中医角度进行症状分析，对可能的疾病与证型进行推断，阐述现有症状如何与这些疾病或证型关联，并解释其辨证依据。
    - 接着，从西医角度进行症状分析，分析可能的疾病，说明症状与疾病之间的病理生理联系。
    - 这一部分中不要出现任何与下一轮问诊问题相关的内容。

2. **引出下一轮问诊的问题**
    - 结合当前症状与初步判断，自然引导出需要进一步询问的【下一轮问诊问题】。分别从中医和西医角度说明询问该问题的目的，例如【下一轮问诊问题】所表明的内容是如何帮助明确证型、鉴别诊断或排除其他病变。
    - 表述逻辑应先说明若出现某种情况可能支持何种判断，表述要丰富（例如，若存在(填症状表现)，在中医上表明**，在西医上表明**）。
    - 前面的分析必须要和引出的【下一轮问诊问题】相关，但是不得出现具体的【下一轮问诊问题】。

3. **输出格式要求**
    - 使用自然语言分段陈述，段落间合理使用衔接词。
    - 严格区分中医与西医分析，避免混杂。
    - 避免使用任何 Markdown 符号（如标题标记、粗体等）。
    - 内容简洁，重点突出，总字数控制在600字以内。
    - 严格按以上顺序与结构进行。
    
请开始生成分析内容：
"""
prompt_v1_1= """/no_think\n假设你是一名中西医结合诊断脾胃病专家，擅长在问诊中系统分析并整理患者症状。请基于【历史问诊信息】和患者最新提供的【下一轮问诊内容】，分析过程与内容遵循以下【思考与分析要求】进行诊断分析与问诊引导。
【历史问诊信息】：{last_round_diagnosis}
【下一轮问诊问题】：{query}

【问诊逻辑与步骤】

核心症状确认：首先锁定患者当前最痛苦、最核心的主诉（如胃痛、反酸、腹胀等），明确其发作时间、性质、程度、持续时间及缓解方式。

病史演变追溯：在明确当前症状后，逐步追问既往病史（如胃炎、溃疡、手术史、长期用药史）、体质特点（如怕冷、易疲劳）及外感病史。

诱因与生活史收集：系统询问可能与症状相关的诱因，包括饮食（辛辣、生冷、饮酒等）、情志（压力、焦虑）、环境（受凉、劳累）及其他因素（如药物服用）。

伴随症状排查：围绕脾胃系统及相关脏腑，询问有无恶心、呕吐、嗳气、食欲变化、排便异常（便秘/腹泻）、黑便、体重下降、口干口苦等全身或局部伴随症状。

信息闭环确认：在收集到一定信息后，主动询问“请问您是否还有其他不适症状？”，若无则进行症状总结与诊断陈述；若有则返回步骤1继续深入。

问诊过程需保持从主到次、由今及昔、从症状到诱因的递进顺序，避免重复或跳跃提问，确保信息收集连贯全面。

【思考与分析要求】
请严格按以下逻辑生成分析内容与下一轮问题：
1. **症状整体描述与分析**
    - 首先，完整分别描述患者已存在的所有症状(如有)和不存在的所有症状(没有不存在的症状就不要加后半句)，若患者有新补充的症状或否定的症状，需明确患者最新补充或否定的内容。
    - 随后，从中医角度进行症状分析，对可能的疾病与证型进行推断，阐述现有症状如何与这些疾病或证型关联，并解释其辨证依据。
    - 接着，从西医角度进行症状分析，分析可能的疾病，说明症状与疾病之间的病理生理联系。
    - 这一部分中不要出现任何与下一轮问诊问题相关的内容。

2. **引出下一轮问诊的问题**
    - 对照【问诊逻辑与步骤】明确指出当前已收集的信息属于“主诉、既往史、生活史、伴随症”中的哪几类，并指出目前缺失的信息类别。
    - 表述逻辑应先说明若出现某种情况可能支持何种判断，表述要丰富（例如，若存在(填症状表现)，在中医上表明**，在西医上表明**）。
    - 逻辑上先说明询问目的，再通过“因此”“接下来需要了解”等衔接，引出【下一轮问诊问题】的问题方向。
    - 前面的分析必须要和引出的【下一轮问诊问题】相关，但是不得出现具体的【下一轮问诊问题】。

3. **输出格式要求**
    - 使用自然语言分段叙述，段落间通过“因此”“由此可见”等连接词衔接。
    - 避免使用任何 Markdown 符号（如标题标记、粗体等）。
    - 生成的格式为{{"analysis":"","question":""}}，不要有其他说明；
    - analysis中生成分析内容，question部分内容包含问题加“主诉、既往史、生活史、伴随症”等中缺失的内容，例如“请问您是否还有其他不适症状？（补充既往史，如既往胃炎、胃溃疡、用药史；生活史，如饮食诱因、情志诱因等）”
    - 内容简介精炼，主题突出，不要说废话,600字以内。
"""

# 到此问诊已结束，请分析并总结所有的患者证症状信息，输出总结内容,不需要再了解患者更多的症状。
# 要求：
# 1.模仿医生问诊的思考辩证过程对患者的所有症状进行分析和总结，分析过程中不需要增加“需进一步了解”的内容；
# 2.分析结束后自然说明问诊结束。
# 3返回格式{{"analysis":"","summarize":""}}，不要有其他说明；
# 4.summarize中内容只对病症进行总结，不要包含分析内容、继续问诊和其他内容。
# 【多轮问诊记录】：\n{multi_round_con}
# 【症状总结】：
# /no_think"""
prompt_v1_2= """/no_think\n假设你是一名中西医结合诊断脾胃病专家，在问诊中擅长系统性地收集和分析患者信息以完善诊断。请你基于【历史问诊信息】进行分析，并根据分析结果得出最终的中医证型和西医疾病的诊断，分析过程与内容遵循以下【思考与分析要求】进行诊断分析与问诊引导。
【历史问诊信息】：{last_round_diagnosis}
【中医证型】：{symptom}
【西医疾病】：{disease}
【思考与分析要求】：
请你严格按照以下步骤和逻辑展开分析，确保推理清晰、因果明确：
1. **症状整体描述与分析**
    - 首先，完整分别描述患者已存在的所有症状(如有)和不存在的所有症状(没有不存在的症状就不要加后半句)，若患者有新补充的症状或否定的症状，需明确患者最新补充或否定的内容。
    - 然后，从中医角度对患者的潜在疾病、证型进行预测和分析，解释症状与预测疾病、证型之间的关联，说明为什么这些症状指向该疾病和证型。
    - 接着，从西医角度对患者的潜在疾病进行预测和分析，解释症状与预测疾病之间的关联，说明为什么这些症状指向该疾病。
    - 最后，提示问诊已结束，解释症状与预测疾病之间的关联，说明为什么这些症状指向该疾病。并分析总结所有的患者证症状信息，输出总结内容,不需要再了解患者更多的症状。
    - 到此问诊已结束，请分析并总结所有的患者证症状信息，输出总结内容,不需要再了解患者更多的症状。最终使分析结果得出【中医证型】和【西医疾病】中描述的诊断结果的中、西医诊断结果。

2. **输出格式要求**
    - 使用自然语言分段叙述，段落间通过“因此”“由此可见”等连接词衔接。
    - 生成的格式为{{"analysis":"","summarize":""}}，不要有其他说明；
    - analysis中生成分析内容，summarize中生成症状总结与中西医诊断(中医证型和西医疾病)，不要生成无关内容。
    - 内容简介精炼，主题突出，不要说废话,600字以内。
    - analysis避免使用任何 Markdown 符号（如标题标记、粗体等）,summarize使用 Markdown 格式，如\n### 症状总结：**\n### 诊断结果：\n#### 中医诊断：**\n#### 西医诊断：**。
"""

prompt_tmp="""/no_think\n作为一名中西医结合诊断的脾胃病专家，请判断【分析内容】中分析的后面部分所描述的‘下一步应关注的问题’或‘进一步应了解的问题’是否与【问题】一致。将判断结果以json格式返回，格式如下：{"result": "一致"/"不一致"}。

【分析内容】：
【问题】：
"""
import json
# 先直接通过多轮问答生成多轮cot指令

def process_item(args):
    line_k, line_data, pbar_queue = args
    pid = os.getpid() # 获取当前进程号
    try:
        # It's better to create the client inside the worker process
        # if it's not picklable or to avoid sharing issues.
        global client
        client = OpenAI(api_key=LLM_API_KEY, base_url=BASE_URL)
        disease_name=line_data['disease_name']
        syndrome_name=line_data['syndrome_name']
        line_conversation = line_data['chat_rounds']['messages']
        if len(line_conversation) < 6:
            pbar_queue.put(1)
            return line_k, None

        line_conversation.append({'role': 'user', 'content': ''})
        expand_cot = {"messages": [{"role": "system", "content": "你是一名中西医结合诊断专家，擅长处理脾胃病领域的疾病诊治。\n请遵循如下规则对患者进行系统化问诊，收集完整的病史信息以辅助诊断。\n问诊规则：\n1.请遵循“从主到次、从现在到既往、从症状到诱因”的递进逻辑进行问诊。\n    (1)主诉优先，聚焦核心症状规律：首先明确患者最痛苦的核心症状，例如胃痛3个月，加重1周，反酸、烧心1个月...等，然后围绕核心症状展开细节追问，包括发作时间（如空腹、餐后、夜间、周期性），症状性质（如胃痛为刺痛、胀痛、隐痛、灼痛，反酸是否伴胸骨后烧灼感），持续时长与缓解方式（如休息后缓解、进食温食缓解、服用抑酸药缓解），症状程度（如轻度、中度、重度，是否影响生活）。\n    (2)由今及昔，追溯病史演变规律:在明确现病史的基础上，逐步追溯既往史（如既往胃炎、胃溃疡、胆囊炎病史）、手术史（如胃肠手术史）、用药史（如长期服用非甾体抗炎药、激素史，中药服用史）；中医需额外关注既往体质（如是否长期怕冷、易疲劳）、外感病史（如近期是否感冒）对脾胃功能的影响。\n    (3)关联诱因，兼顾内外因素规律:全面追问脾胃病发作的相关内外因素，包括饮食诱因（如暴饮暴食、辛辣刺激、生冷饮食、不洁饮食、饮酒、浓茶咖啡摄入）、情志诱因（如近期压力大、焦虑、生气、熬夜）、环境与劳累诱因（如受凉、过度劳累）、其他诱因（如服用特定药物后发作）。\n    (4)系统关联，排查合并症状规律：脾胃与其他脏腑关联密切，需排查相关系统伴随症状，例如消化系统关联症状（如恶心呕吐、腹胀、嗳气、食欲不振、便秘、腹泻、黑便、便血、黄疸）、全身伴随症状（如体重下降、乏力、贫血、发热）、中医特殊伴随症状（如口干口苦、口淡无味、口臭、喜食热饮/冷饮、大便黏滞、小便黄赤/清长）。\n2.问诊过程需保持逻辑连贯与层次递进，避免重复问诊及跨阶段跳跃式提问，以保障信息收集的全面性、系统性和可用性。\n3.当收集到足够多患者症状信息后询问“请问您是否还有其他不适症状？”，若患者回答其他症状，则重新进行1.中的问诊；若患者回答无，则进行总结陈述，内容包括：\n    (1)症状总结：简要概括患者的主要症状、持续时间及相关特点。\n    (2)诊断结果：基于收集的病史信息，给出中医证型和西医诊断。"}]}
        expand_cot1={'messages':[{"role": "system", "content": "你是一名中西医结合诊断专家，擅长处理脾胃病领域的疾病诊治。\n请遵循如下规则对患者进行系统化问诊，收集完整的病史信息以辅助诊断。\n问诊规则：\n1.请遵循“从主到次、从现在到既往、从症状到诱因”的递进逻辑进行问诊。\n    (1)主诉优先，聚焦核心症状规律：首先明确患者最痛苦的核心症状，例如胃痛3个月，加重1周，反酸、烧心1个月...等，然后围绕核心症状展开细节追问，包括发作时间（如空腹、餐后、夜间、周期性），症状性质（如胃痛为刺痛、胀痛、隐痛、灼痛，反酸是否伴胸骨后烧灼感），持续时长与缓解方式（如休息后缓解、进食温食缓解、服用抑酸药缓解），症状程度（如轻度、中度、重度，是否影响生活）。\n    (2)由今及昔，追溯病史演变规律:在明确现病史的基础上，逐步追溯既往史（如既往胃炎、胃溃疡、胆囊炎病史）、手术史（如胃肠手术史）、用药史（如长期服用非甾体抗炎药、激素史，中药服用史）；中医需额外关注既往体质（如是否长期怕冷、易疲劳）、外感病史（如近期是否感冒）对脾胃功能的影响。\n    (3)关联诱因，兼顾内外因素规律:全面追问脾胃病发作的相关内外因素，包括饮食诱因（如暴饮暴食、辛辣刺激、生冷饮食、不洁饮食、饮酒、浓茶咖啡摄入）、情志诱因（如近期压力大、焦虑、生气、熬夜）、环境与劳累诱因（如受凉、过度劳累）、其他诱因（如服用特定药物后发作）。\n    (4)系统关联，排查合并症状规律：脾胃与其他脏腑关联密切，需排查相关系统伴随症状，例如消化系统关联症状（如恶心呕吐、腹胀、嗳气、食欲不振、便秘、腹泻、黑便、便血、黄疸）、全身伴随症状（如体重下降、乏力、贫血、发热）、中医特殊伴随症状（如口干口苦、口淡无味、口臭、喜食热饮/冷饮、大便黏滞、小便黄赤/清长）。\n2.问诊过程需保持逻辑连贯与层次递进，避免重复问诊及跨阶段跳跃式提问，以保障信息收集的全面性、系统性和可用性。\n3.当收集到足够多患者症状信息后询问“请问您是否还有其他不适症状？”，若患者回答其他症状，则重新进行1.中的问诊；若患者回答无，则进行总结陈述，内容包括：\n    (1)症状总结：简要概括患者的主要症状、持续时间及相关特点。\n    (2)诊断结果：基于收集的病史信息，给出中医证型和西医诊断。"}]}
        # sys assi user assi user assi user ...
        for i, (q, a) in enumerate(zip(line_conversation[::2], line_conversation[1::2])):
            flag=False
            if i == 0:
                # assistant_content = "<think>\n我是从事脾胃病诊疗的专家，主要擅长从中西医结合角度处理各种脾胃相关疾病，包括胃痛、反酸、嗳气、腹胀、食欲异常、腹泻或便秘等情况。\n\n脾胃在中医里被称为“后天之本”，一旦运化功能出现问题，身体其他系统往往也会受到影响。所以在真正下诊断、谈治疗之前，最关键的一步不是判断病名，而是把症状问清楚、问细致。\n\n在临床上，同样的症状在不同患者身上，病因、病机和处理思路可能完全不同。比如胃痛，既可能与胃酸、黏膜损伤有关，也可能涉及寒热虚实、气机失调。\n脾胃病问诊的难点就在于：\n\n- 症状往往反复、变化多\n\n- 患者主观感受差异很大\n\n- 表现相似，但证型可能完全不同\n\n因此，我会通过循序渐进的方式，从最主要的不适开始，一点点了解您的症状特点、变化规律以及伴随情况，帮助判断真正的病机所在。\n\n下面我们正式开始问诊。\n</think>\n\n请您先说说，目前最让您难受、最想解决的症状是什么？比如：胃痛、反酸、腹胀、恶心、腹泻等。"
                # expand_cot['messages'].append({"role": "assistant", "content": assistant_content})
                expand_cot['messages'].append({"role": "user", "content": a['content']})
                expand_cot1['messages'].append({"role": "user", "content": a['content']})
            elif i > 0 and i < len(line_conversation) // 2 - 2:
                query_tendency = q['content']
                prompt = prompt_v1.format(last_round_diagnosis=json.dumps(expand_cot1['messages'][1:], ensure_ascii=False), query=query_tendency)
                # print("================================")
                # print(prompt)
                
                max_check_retries = 3
                check_retry_count = 0
                while not flag and check_retry_count < max_check_retries:
                    resp = llm_generation_and_check(prompt)
                    
                    resp_str = resp.replace("<think>", "").replace("</think>", "").strip()

                    
                    # 判断cot与query是否一致
                    prompt_judge=prompt_tmp.replace("【分析内容】：", f"【分析内容】：{resp_str}").replace("【问题】：", f"【问题】：{q['content']}")
                    resp_judge=llm_generation_and_check(prompt_judge)
                    resp_judge_json=clean_response_to_json(resp_judge)
                    if resp_judge_json.get("result","不一致")=="不一致":
                        check_retry_count += 1
                        print(f"[PID:{pid}] Item {line_k}: Consistency check failed. Retry {check_retry_count}/{max_check_retries}...")
                        pass
                    else:
                        flag=True
                
                if not flag:
                    print(f"[PID:{pid}] Item {line_k}: Failed to get consistent response after {max_check_retries} retries. Skipping.")
                    pbar_queue.put(1)
                    return line_k, None
                
                resp_str = "<think>\n" + resp_str + "\n</think>\n\n" + query_tendency
                expand_cot['messages'].append({"role": "assistant", "content": resp_str})
                expand_cot['messages'].append({"role": "user", "content": a['content']})
                #
                expand_cot1['messages'].append({"role": "assistant", "content": query_tendency})
                expand_cot1['messages'].append({"role": "user", "content": a['content']})
            elif i == len(line_conversation) // 2 - 2:
                query_tendency = q['content']
                prompt = prompt_v1_1.format(last_round_diagnosis=json.dumps(expand_cot1['messages'][1:], ensure_ascii=False), query=query_tendency)
                
                max_check_retries = 3
                check_retry_count = 0
                while not flag and check_retry_count < max_check_retries:
                    resp = llm_generation_and_check(prompt)
                    resp_json = clean_response_to_json(resp)
                    resp_cot = resp_json.get("analysis")
                    resp_question = resp_json.get('question')

                    if resp_cot is None or resp_question is None:
                        print(f"[PID:{pid}] Item {line_k}: LLM response missing 'analysis' or 'question'. Retrying...")
                        check_retry_count += 1
                        continue
                    
                    # 判断cot与query是否一致
                    prompt_judge=prompt_tmp.replace("【分析内容】：", f"【分析内容】：{resp_cot}").replace("【问题】：", f"【问题】：{resp_question}")
                    resp_judge=llm_generation_and_check(prompt_judge)
                    resp_judge_json=clean_response_to_json(resp_judge)
                    if resp_judge_json.get("result","不一致")=="不一致":
                        check_retry_count += 1
                        print(f"[PID:{pid}] Item {line_k}: Consistency check failed. Retry {check_retry_count}/{max_check_retries}...")
                        pass
                    else:
                        flag=True

                if not flag:
                    print(f"[PID:{pid}] Item {line_k}: Failed to get consistent response after {max_check_retries} retries. Skipping.")
                    pbar_queue.put(1)
                    return line_k, None
                
                resp_str = "<think>\n" + resp_cot + "\n</think>\n\n" + resp_question
                expand_cot['messages'].append({"role": "assistant", "content": resp_str})
                expand_cot['messages'].append({"role": "user", "content": a['content']})
                
                #
                expand_cot1['messages'].append({"role": "assistant", "content": resp_json['question']})
                expand_cot1['messages'].append({"role": "user", "content": a['content']})
            else:
                prompt = prompt_v1_2.format(last_round_diagnosis=json.dumps(expand_cot1['messages'][1:], ensure_ascii=False),disease=disease_name,symptom=syndrome_name)
                
                max_check_retries = 3
                check_retry_count = 0
                while not flag and check_retry_count < max_check_retries:
                    resp = llm_generation_and_check(prompt)
                    resp_json = clean_response_to_json(resp)
                    resp_cot = resp_json.get("analysis")
                    resp_summarize = resp_json.get('summarize')

                    if resp_cot is None or resp_summarize is None:
                        print(f"[PID:{pid}] Item {line_k}: LLM response missing 'analysis' or 'summarize'. Retrying...")
                        check_retry_count += 1
                        continue
                    
                    # 判断cot与query是否一致
                    prompt_judge=prompt_tmp.replace("【分析内容】：", f"【分析内容】：{resp_cot}").replace("【问题】：", f"【问题】：{resp_summarize}")
                    resp_judge=llm_generation_and_check(prompt_judge)
                    resp_judge_json=clean_response_to_json(resp_judge)
                    if resp_judge_json.get("result","不一致")=="不一致":
                        check_retry_count += 1
                        print(f"[PID:{pid}] Item {line_k}: Consistency check failed. Retry {check_retry_count}/{max_check_retries}...")
                        pass
                    else:
                        flag=True

                if not flag:
                    print(f"[PID:{pid}] Item {line_k}: Failed to get consistent response after {max_check_retries} retries. Skipping.")
                    pbar_queue.put(1)
                    return line_k, None
                
                resp_str = "<think>\n" + resp_json["analysis"] + "\n</think>\n\n" + resp_json['summarize']
                expand_cot['messages'].append({"role": "assistant", "content": resp_str})
                expand_cot['messages'].append({"role": "user", "content": a['content']})
                
                #
                expand_cot1['messages'].append({"role": "assistant", "content": resp_json['summarize']})
                expand_cot1['messages'].append({"role": "user", "content": a['content']})

        expand_cot['messages'] = expand_cot['messages'][:-1]
        last_content = expand_cot['messages'][-1]
        last_content_str = last_content['content']
        last_content_str = last_content_str.replace("</think>\n\n", "</think>\n\n问诊结束，结果如下：\n")
        last_content_json = {"role": "assistant", "content": last_content_str}
        line_save = expand_cot['messages'][:-1]
        line_save.append(last_content_json)
        expand_cot['messages'] = line_save
        
        pbar_queue.put(1)
        return line_k, expand_cot
    except Exception as e:
        print(f"Error processing item {line_k}: {e}")
        pbar_queue.put(1)
        return line_k, None

def pbar_updater(pbar_queue, total):
    pbar = tqdm(total=total)
    while pbar.n < total:
        try:
            pbar.update(pbar_queue.get(timeout=10))
        except Exception:
            pass
    pbar.close()

if __name__ == '__main__':
    # ============================================================
    # Input/output paths (override via environment variables)
    #   export INPUT_DIALOGUE_PATH=/path/to/input.json
    #   export OUTPUT_COT_PATH=/path/to/output.jsonl
    # ============================================================
    file_multi_round = os.getenv("INPUT_DIALOGUE_PATH", "./data/input_dialogue.json")
    file_save = os.getenv("OUTPUT_COT_PATH", "./data/output_cot.jsonl")
    
    with open(file_multi_round, 'r') as f:
        lines = json.load(f)

    # 跳过前*154个，继续处理剩余的
    lines = {k: v for k, v in lines.items() if int(k) >= 154}
    
    
    # # 如果输出文件已存在，先删除，确保从头开始写入
    # if os.path.exists(file_save):
    #     os.remove(file_save)

    manager = Manager()
    pbar_queue = manager.Queue()
    
    # 按原始顺序准备任务参数
    tasks = [(key, lines[key], pbar_queue) for key in sorted(lines.keys())]
    
    # 启动进度条更新进程
    pbar_process = Pool(1).apply_async(pbar_updater, (pbar_queue, len(tasks)))

    # 使用 with open 确保文件最终能被正确关闭
    with open(file_save, 'a', encoding='utf-8') as f_out:
        with Pool(processes=8) as pool:
            # 使用 imap 来保持顺序并逐个处理结果
            # 这样可以实现“处理一条，写入一条”
            for key, result_data in pool.imap(process_item, tasks):
                if result_data:
                    # 将处理好的单条结果立即写入文件
                    f_out.write(json.dumps(result_data, ensure_ascii=False) + "\n")

    pbar_process.wait()
    print("Processing completed.")