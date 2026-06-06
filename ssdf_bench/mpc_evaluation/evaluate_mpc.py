import json
import argparse
import os
import re
from openai import OpenAI

import json_repair
from typing import Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Eval:
    def __init__(self, eval_file, base_url, model):
        
        # NOTE: Some local OpenAI-compatible backends ignore api_key, so a
        # placeholder is accepted in that case. For cloud APIs, set api_key
        # via environment variable OPENAI_API_KEY.
        api_key = os.getenv("OPENAI_API_KEY", "-")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model=model
        
        self.eval_file = eval_file
        self.examed_data = self._load_eval_data()

        self.effective_consultation_list=[] #有效问诊长度
        self.predicted_symptoms_list = [] #预测症状list
        
        # self.deny_list=["不","没","无","尚"]
        self._split_symptoms_prompt="""/no_think\n你是一名中西医结合诊断专家。请将以下【患者症状描述】拆分为最小单元的症状列表。
拆分规则：
1.将复合症状拆分为不可再分的最小具体症状单元。示例：“发热恶寒” → “发热”、“恶寒”；
2.严格依据【患者症状描述】原文，不得虚构、增减、删减任何症状；
3.仅列出患者明确存在的具体症状，不包含患者不存在的症状、阴性体征、无异常等内容。
4.将同义或描述相近的症状合并，确保症状列表中出现症状的唯一性。
示例：“咳痰”与“咳嗽有痰” → 统一为“咳痰”；“胸骨后持续性灼热感”与“胸骨后间歇性灼热感” → 统一为“胸骨后灼热”。
6.仅返回如下 JSON 格式，不要添加任何额外说明、标题或注释：
{{"symptoms_list": ["症状1", "症状2", "症状3", ...]}}

# 【患者症状描述】：{synptom_desc}""" 

    def _load_eval_data(self):
        try:
            with open(self.eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            return eval_data
        except FileNotFoundError:
            print(f"错误：评估文件 {self.eval_file} 未找到。")
            return None
        except json.JSONDecodeError:
            print(f"错误：无法解析JSON文件 {self.eval_file}。")
            return None
    
    def clean_response_to_json(self,text: str) -> Any:
        """
        Clean LLM output text and parse it as JSON (enhanced).
        Can automatically repair unescaped quotes, line breaks, and Markdown noise.
        Supports parsing either a single object {} or an array [].
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

    def _llm(self, prompt, system="You are a helpful assistant."):
        messages=[]
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=0.0,
        max_tokens=8192,
        stream=False,
        ) 
        return response.choices[0].message.content
    
    def  llm_generation_and_check(self,prompt,type=str, system="You are a helpful assistant."):
        resp_str = None
        retry_times = 0
        max_retries = 5

        while not isinstance(resp_str, type) and retry_times < max_retries:
            try:
                resp_str = self._llm(prompt,system)
                
            except Exception as e:
                retry_times += 1
                print(f"llm_judge generation error.Retry {retry_times}/{max_retries} times, due to error: {e}")

        # If you still fail after multiple attempts, return an empty string to avoid subsequent split errors
        if not isinstance(resp_str, type):
            return ""
        return resp_str

    def _parse_llm_symptoms(self, last_assistant_message):
        """
        从LLM的最后一次回复中解析出症状总结。
        """
        # 症状总结
        symptom_summary_match = re.search(
            r"症状总结\s*[:：*】\]]\s*(.*?)(?=\s*(?:诊断结果|【|诊断分析|二、|\Z))",
            last_assistant_message,
            re.DOTALL
        )
        symptom_summary = (
            symptom_summary_match.group(1).strip().strip("#").strip()
            if symptom_summary_match
            else ""
        )

        return symptom_summary

    def _abstract_model_symptoms(self,chat_rounds):
        """
        Given a consultation record and model response, a summary of symptoms is extracted.
        """    

        if not chat_rounds or len(chat_rounds) == 0:
            print(f"Warning: The record is missing 'exam_chat_rounds' and will be skipped.")
            return  []
            
        last_message = chat_rounds[-1]
        if last_message.get("role") == "assistant":
            content = last_message.get("content", "")
            symptoms_str = self._parse_llm_symptoms(content)
            

            prompt=self._split_symptoms_prompt.format(synptom_desc=symptoms_str)
            resp=self.llm_generation_and_check(prompt, type=str)
            resp_json=self.clean_response_to_json(resp)
            symptoms_list=resp_json.get("symptoms_list", []) if isinstance(resp_json, dict) else []
        else:
          
            symptoms_list=[]
        return  symptoms_list

    def diagnostic_Accuracy(self, predicted_tcm,predicted_western,ground_truth_tcm,ground_truth_western):
        """
        # 1.Diagnostic Accuracy
        # The diagnostic conclusions given by the model are compared with the diagnostic labels in the standard medical record (Gold Standard).
        # Use the Macro-F1 Score
        # $$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
        # Pass in the list directly to calculate the average score
        
        Parameter examples:
         # Simulate real tags
    ground_truth_western = [
        "gastritis", "gastric ulcer", "functional dyspepsia", "gastritis", "gastric ulcer"
    ]

    predicted_western = [
        "gastritis", "gastritis", "functional dyspepsia", "gastritis", "gastric ulcer"
    ]

    ground_truth_tcm = [
        "Spleen and stomach deficiency and cold", "Liver and stomach disharmony", "Damp heat connotation", "Spleen and stomach deficiency and cold", "Qi stagnation and blood stasis"
    ]

    predicted_tcm = [
        "Spleen and stomach deficiency and cold", "Liver and stomach disharmony", "Damp and heat connotation", "Damp heat connotation", "Qi stagnation and blood stasis"
    ]
        # """
         # 西医诊断评估
        print("ground_truth_western")
        print(ground_truth_western)
        print("predicted_western")
        print(predicted_western)
        western_accuracy = accuracy_score(ground_truth_western, predicted_western)
        western_precision, western_recall, western_f1, _ = precision_recall_fscore_support(ground_truth_western, predicted_western, average='weighted', zero_division=0)
        
        print("\n--- 西医诊断评估 ---")
        print(f"样本总数: {len(ground_truth_western)}")
        print(f"准确率 (Accuracy): {western_accuracy:.4f}")
        print(f"加权平均精确率 (Precision): {western_precision:.4f}")
        print(f"加权平均召回率 (Recall): {western_recall:.4f}")
        print(f"加权平均 F1-Score: {western_f1:.4f}")

        # 中医证型评估
        tcm_accuracy = accuracy_score(ground_truth_tcm, predicted_tcm)
        tcm_precision, tcm_recall, tcm_f1, _ = precision_recall_fscore_support(ground_truth_tcm, predicted_tcm, average='weighted', zero_division=0)

        print("\n--- 中医证型评估 ---")
        print(f"样本总数: {len(ground_truth_tcm)}")
        print(f"准确率 (Accuracy): {tcm_accuracy:.4f}")
        print(f"加权平均精确率 (Precision): {tcm_precision:.4f}")
        print(f"加权平均召回率 (Recall): {tcm_recall:.4f}")
        print(f"加权平均 F1-Score: {tcm_f1:.4f}")

        # # 打印一些预测错误的例子
        # print("\n--- 部分错误案例分析 ---")
        # error_count = 0
        # for i in range(len(ground_truth_western)):
        #     if (predicted_western[i] != ground_truth_western[i] or predicted_tcm[i] != ground_truth_tcm[i]) and error_count < 5:
        #         print(f"\n案例 {i+1}:")
        #         print(f"  患者医案: {self.examed_data[i].get('patient_summary')}")
        #         print(f"  标准西医诊断: {ground_truth_western[i]}")
        #         print(f"  预测西医诊断: {predicted_western[i]}")
        #         print(f"  标准中医证型: {ground_truth_tcm[i]}")
        #         print(f"  预测中医证型: {predicted_tcm[i]}")
        #         error_count += 1
        return (tcm_accuracy,tcm_precision, tcm_recall, tcm_f1), (western_accuracy,western_precision, western_recall, western_f1)   

    def recall_must(self,ground_truth_key_symptoms, predicted_symptoms):
        """
        Calculate critical symptom recall
        
        Args:
            ground_truth_key_symptoms: A list of key symptoms included in the medical record
            predicted_symptoms: List of symptoms predicted/induced by the model
        
        Returns:
            recall_score: Recall (between 0-1)
            details: Matching details
            
        Use Cases:
            # # Single medical record calculation
            # result1 = recall_must(key_symptoms_1, predicted_1)
            # print(f"Recall: {result1['recall']}") # 0.25 (only matches to "fever")
            # print(f"Missing symptom: {result1['missed_symptoms']}") # ['cough', 'fatigue', 'Nacha']

        """
        # Calculate the number of key symptoms successfully induced (intersection)
        matched = set(ground_truth_key_symptoms) & set(predicted_symptoms)
        matched_count = len(matched)
        
        # The total number of key symptoms included in the medical record
        total_count = len(ground_truth_key_symptoms)
        
        recall = matched_count / total_count if total_count > 0 else 0
        
        return {
            "recall": round(recall, 4),
            "matched_symptoms": list(matched),
            "missed_symptoms": list(set(ground_truth_key_symptoms) - set(predicted_symptoms)),
            "matched_count": matched_count,
            "total_count": total_count
        }
    
    def batch_Information_Recall(self, cases):
        """
        # Proactivity & Information Recall
        # $Recall_{must} = \frac{\text{Number of key symptoms successfully induced by the model}}{\text{Total number of key symptoms in the medical record}}
        Calculate key symptom recall across multiple medical records in bulk
        
            cases: list, each element (medical record key symptom list, model predicted symptom list)
        Args:
        
            Overall recall rate and details of each medical record
        # Batch calculations

        Returns:
        cases = [(key_symptoms_1, predicted_1), (key_symptoms_2, predicted_2)]
        batch_result = self.batch_Information_Recall(cases)
        Use Cases:
        print(f"Overall recall: {batch_result['overall_recall']}")
        """
        total_matched = 0
        total_key = 0
        results = []
        
        for i, (key_symptoms, pred_symptoms) in enumerate(cases):
            result = self.recall_must(key_symptoms, pred_symptoms)
            results.append({
                "case_id": i + 1,
                **result
            })
            total_matched += result["matched_count"]
            total_key += result["total_count"]
        
        overall_recall = total_matched / total_key if total_key > 0 else 0
        
        return {
            "overall_recall": round(overall_recall, 4),
            "total_matched": total_matched,
            "total_key": total_key,
            "case_details": results
        }
    
    def extract_questions(self,text: str):
        pattern = r'[^。！？\n]*[^?？\s][?？]'
        matches = re.findall(pattern, text)
        return [q.strip() for q in matches]
#     def diagnose_efficiency(self,dialog_len_list,ground_truth_symptoms_list,predicted_symptoms_list):
#         """#  Dialogue Efficiency
#         Conversation Efficiency:
#         Efficiency = Number of valid symptoms / Number of conversation rounds T

#         Back:
#             mean_efficiency: Average efficiency across all samples
#             efficiency_list: Efficiency per sample
            
# Reference examples:
# Input:
# dialog_len_list = [6, 8]

# ground_truth_symptoms_list = [
#     ["acid reflux", "heartburn", "belching"],
#     ["abdominal pain", "bloating"]
# ]

# predicted_symptoms_list = [
#     ["acid reflux", "belching", "nausea"],
#     ["Abdominal pain"]
# ]
# diagnose_efficiency(dialog_len_list, ground_truth_symptoms_list, predicted_symptoms_list)

# Back example
# (0.22916666666666666, [0.3333333333333333, 0.125])
# """    
#         assert len(dialog_len_list) == len(ground_truth_symptoms_list) == len(predicted_symptoms_list)

#         efficiency_list = []

#         for T, gt_symptoms, pred_symptoms in zip(
#             dialog_len_list,
#             ground_truth_symptoms_list,
#             predicted_symptoms_list
#         ):
#             if T == 0:
#                 efficiency_list.append(0.0)
#                 continue

#             # 有效症状数 = 预测和真实的交集
#             valid_symptoms = set(gt_symptoms) & set(pred_symptoms)
#             valid_count = len(valid_symptoms)

#             efficiency = valid_count / T
#             efficiency_list.append(efficiency)

#         mean_efficiency = sum(efficiency_list) / len(efficiency_list) if efficiency_list else 0.0

#         return mean_efficiency, efficiency_list

    def calculate_effective_consultation_num(self,exam_chat_rounds:List):
        effective_consultation=0
        for q,a in zip(exam_chat_rounds[::2],exam_chat_rounds[1::2]):
            if a['content'][:1] in self.deny_list:
                pass
            else:
                effective_consultation+=1
                
        return effective_consultation  
def main():
    parser = argparse.ArgumentParser(description="To evaluate the effect of multiple rounds of consultation in the spleen and stomach disease model.")

    parser.add_argument("--eval_file", type=str, default="./results/consultation_results.json",
                      help="Path to the experimental results JSON file to be evaluated.")
    parser.add_argument("--model_name", type=str, default="Qwen3-32B",
                      help="The name or ID of the model used for evaluation.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                      help="The base URL of the model service.")
    parser.add_argument("--output_file", type=str, default="./results/evaluation_results.json",
                      help="Path for the evaluation results output.")
    parser.add_argument("--model", type=str, default="Qwen3-14B", help="")
    args = parser.parse_args()
    evaluator = Eval(args.eval_file, args.base_url, args.model_name)


    
    dialog_len_list=[]

    
    for i,item in enumerate(evaluator.examed_data):
        print(i)
        
        effective_consultation=item['effective_consultation']
        evaluator.effective_consultation_list.append(effective_consultation)
        
        exam_chat_rounds=item.get("exam_chat_rounds", [])
        dialog_len=0
        for msg in exam_chat_rounds:
            if msg.get("role")=="assistant":
                question_list=evaluator.extract_questions(msg.get("content", ""))
                dialog_len+=len(question_list)
        dialog_len=max(dialog_len,int(len(exam_chat_rounds)/2))
        dialog_len_list.append(dialog_len)

        # Calculating the number of valid consultations in test data
        symptoms_list=evaluator._abstract_model_symptoms(exam_chat_rounds)
        evaluator.predicted_symptoms_list.append(symptoms_list)
        


    
    print("任务1 "+"="*20)
    # Consultation effectiveness
    if len(evaluator.effective_consultation_list)!=len(evaluator.predicted_symptoms_list):
        raise f"The lengths of effective_consultation_list({len(evaluator.effective_consultation_list)}) effective_consultation_list_exam_data({len(evaluator.predicted_symptoms_list)})  are not equal."
    
    effective_consultation_ratio=sum([len(s) for s in evaluator.predicted_symptoms_list])/sum(evaluator.effective_consultation_list)
    
    # effective_consultation_ratio2=sum([j/k for j,k in zip([len(s) for s in evaluator.predicted_symptoms_list], evaluator.effective_consultation_list) ])/len(evaluator.effective_consultation_list)
    
    print(f"Number of valid consultations for the test model:{[len(s) for s in evaluator.predicted_symptoms_list]}")
    print(f"Number of valid consultations for test questions:{evaluator.effective_consultation_list}")
    print(f"Total number of valid symptoms obtained by the test model:{sum([len(s) for s in evaluator.predicted_symptoms_list])}, Total number of symptoms in exam datas:{sum(evaluator.effective_consultation_list)}")
    print(f"Information collection rate:{effective_consultation_ratio}")
    # print(f"信息收集率2：{effective_consultation_ratio2}")
    
    print("任务2 "+"="*20)
    # Dialogue Efficiency
    # $$\text{Efficiency} = \frac{\text{Number of valid symptoms obtained}}{\text{Total Conversation Rounds } T}$$
    # Calculate each one and then find the average
    if len(dialog_len_list)!=len(evaluator.predicted_symptoms_list):
        raise f"The lengths of dialog_len_list({len(dialog_len_list)}) predicted_symptoms_list({len(evaluator.predicted_symptoms_list)})  are not equal."
    
    dialogue_efficiency=sum([j/k for j,k in zip([len(s) for s in evaluator.predicted_symptoms_list], dialog_len_list) ])/len(dialog_len_list)
    
    print(f"Number of valid consultations for the test model:{[len(s) for s in evaluator.predicted_symptoms_list]}")
    print(f"Length of consultations for test questions:{dialog_len_list}")
    print(f"Dialogue efficiency:{dialogue_efficiency}")
    
    rest_all={

        "effective_consultation_list":evaluator.effective_consultation_list,
        "effective_consultation_list_len":len(evaluator.effective_consultation_list),
        "predicted_symptoms_list":evaluator.predicted_symptoms_list,
        "predicted_symptoms_list_len":len(evaluator.predicted_symptoms_list),
        "dialog_len_list":dialog_len_list,
        "test_model_symptoms":sum([len(s) for s in evaluator.predicted_symptoms_list]),
        "exam_datas_symptoms":sum(evaluator.effective_consultation_list),
        "effective_consultation_ratio":effective_consultation_ratio,
        "dialogue_efficiency":dialogue_efficiency,

    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(rest_all, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()  