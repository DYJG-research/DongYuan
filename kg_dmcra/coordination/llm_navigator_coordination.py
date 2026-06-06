import logging

from pathlib import Path
import requests
import json
import time
import sys
from openai import OpenAI
import pandas as pd
import os
from typing import List
import re
import json_repair

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class SSDFConsultationClient:
    """SSDF collaborative consultation client that exposes an OpenAI-compatible API"""

    def __init__(self,
                ssdf_core_model="test",
                ssdf_core_model_base_url="",
                ssdf_core_sys_prompt="",
                history_length=5,
                low_threshold=0.1,
                final_low_threshold=0.6,  
                max_retry=3,
                navigator_url="",
                navigator_topk=3,
                symptom_file="",
                question_cleaned_file=""
                ):
        
        # Standard symptom list
        self.standard_symptoms_list = pd.read_csv(symptom_file)['symptom'].tolist()  # Example standard symptom list

        # Load symptom-to-question mapping
        self.symptom_to_question=self._get_symptom_to_question(question_cleaned_file)
        # Load question-to-symptom mapping
        self.question_to_symptom=self._get_question_to_symptom(question_cleaned_file)
        
        # SSDF-core-model parameters
        self.ssdf_core_model=ssdf_core_model
        self.ssdf_core_model_base_url=ssdf_core_model_base_url
        # Initialize OpenAI client (backbone model)
        # NOTE: Some local OpenAI-compatible backends ignore api_key, so a
        # placeholder is accepted in that case. For cloud APIs, set api_key
        # via environment variable OPENAI_API_KEY or pass it to the constructor.
        api_key = os.getenv("OPENAI_API_KEY", "-")
        self.ssdf_core_client = OpenAI(api_key=api_key, base_url=self.ssdf_core_model_base_url)
        self.ssdf_core_sys_prompt=ssdf_core_sys_prompt
        self.ssdf_core_messages=[{"role": "system", "content": self.ssdf_core_sys_prompt}]
        
        # SSDF-Navigator parameters
        self.navigator_url = navigator_url  # Navigator API URL
        self.navigator_topk = navigator_topk  # Number of top-K symptoms returned by the Navigator
        
        # Queue parameters
        self.history_length = history_length  # Queue length L for the symptom history
        self.low_threshold = low_threshold  # Minimum probability threshold
        self.final_low_threshold = final_low_threshold # Higher threshold used when the LLM misses Navigator suggestions three times
        self.max_retry = max_retry  # Maximum retry count
        
        # Length-L queue of historical question symptoms
        self.symptom_history = []
        # Track asked questions to avoid duplicates
        self.asked_questions = []
        logger.info("Initializing SSDFConsultationClient - Core model: {}, Navigator URL: {}, history queue length: {}, threshold: {}, max retries: {}".format(
            self.ssdf_core_model, self.navigator_url, self.history_length, self.low_threshold, self.max_retry))
    def _get_symptom_to_question(self, file_path: str):
        # Build the symptom-to-question mapping (use the first question for each symptom)
        symptom_to_question={}
        question_mapping = pd.read_csv(file_path)
        for _, row in question_mapping.iterrows():
            symptom = row['symptom']
            question = row['result']
            if symptom not in symptom_to_question:
                symptom_to_question[symptom] = question
        return symptom_to_question
    
    def _get_question_to_symptom(self,file_path:str):
        question_to_symptom={}
        question_mapping = pd.read_csv(file_path)
        for _, row in question_mapping.iterrows():
            symptom = row['symptom']
            question = row['result']
            if symptom not in question_to_symptom:
                question_to_symptom[question] = symptom
        return question_to_symptom
        
    def get_question_for_symptom(self, symptom: str) -> str:
        """Return the question text that corresponds to a given symptom"""
        return self.symptom_to_question.get(symptom, f"您是否有{symptom}的症状？")

    def _navigator_predict(self, symptom_history:List[str], topk=3):
        # SSDF-Navigator API invocation
        # Request example:
        # curl -X POST "http://localhost:8999/predict" \
        #      -H "Content-Type: application/json" \
        #      -d '{"symptoms": ["腹痛", "腹胀"], "topk": 3}'
        
        # Response format:
        #   {'predicted_symptom': '小便短赤',
        #  'topk_symptoms': ['小便短赤', '恶心或呕吐', '口干或口苦'],
        #  'topk_probabilities': [0.34184515476226807,
        #   0.27840131521224976,
        #   0.08028032630681992],
        #  'history_before_focus': [],
        #  'current_focus': '腹痛'}
        url = self.navigator_url
        payload = {
            "symptoms": symptom_history,
            "topk": topk
        }
        headers = {
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            if response.status_code == 200:
                result = response.json()
                predicted_symptom = result.get("predicted_symptom", "")
                topk_symptoms = result.get("topk_symptoms", [])
                topk_probabilities = result.get("topk_probabilities", [])

                # Build a symptom-to-probability mapping
                symptom_prob_dict = {}
                if topk_symptoms and topk_probabilities:
                    # Ensure the two lists have the same length
                    try:
                        (len(topk_symptoms) == len(topk_probabilities))
                    except Exception as e:
                        raise ValueError(f"Navigator API returned mismatched lengths for topk_symptoms and topk_probabilities: {e}")
                    
                    for symptom, probability in zip(topk_symptoms, topk_probabilities):
                        symptom_prob_dict[symptom] = probability

                return predicted_symptom, topk_symptoms, topk_probabilities, symptom_prob_dict
            else:
                print(f"Error: Received status code {response.status_code} from navigator API")
                return "", [], [], {}
        except requests.exceptions.RequestException as e:
            print(f"Navigator API connection error: {e}")
            print(f"Please verify that the Navigator service is running at {url}")
            return "", [], [], {}
   
    def _generate_diagnose_result(self,  messages):
        """Generate the final diagnosis output when the consultation ends proactively"""
        diagnose_records=""
        for row in messages:
            if row["role"]=="assistant" or row["role"]=="user":
                    diagnose_records=diagnose_records+"[医生]："+row["content"]+"\n" if row["role"]=="assistant" else "[患者]："+row["content"]+"\n"
            
        prompt="""/no_think\n请根据【问诊记录】问诊内容总结诊断结论，明确中医证型和西医诊断。
    【问诊记录】：{diagnose}
    要求：
        1.直接返回最终结果，不添加其他内容。
        2.返回格式如下：
            问诊结束，结果如下：
            ### 症状总结：填写总结患者的所有症状表现。
            
            ### 诊断结果：
                #### 中医诊断：填写中医证型诊断结果。
            
                #### 西医诊断：填写西医诊断结果。
        
                """.format(diagnose=diagnose_records)
                
        response = self.ssdf_core_client.chat.completions.create(
                    model=self.ssdf_core_model,
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8192
                )
        return response.choices[0].message.content
    
    def _core_llm(self, messages, system="You are a helpful assistant."):
        "SSDF-core-DPO large-language-model invocation interface"
        # If a custom system prompt is provided, prefer it; otherwise fall back to the default
        # instruction set that guides the doctor model through structured gastroenterology
        # consultations (chief complaint focus, history tracing, trigger exploration, and
        # associated symptom screening).
 
        # messages=[]
        # messages.append({"role": "system", "content": system})
        self.ssdf_core_messages=messages
        try:
            response = self.ssdf_core_client.chat.completions.create(
            model=self.ssdf_core_model,
            messages=self.ssdf_core_messages,
            temperature=0.0,
            max_tokens=8192,
            stream=False,
                        # Avoid passing tools to prevent accidental tool invocations that alter response formats
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed: {e}")
            print(f"Please verify that the LLM service is running at {self.ssdf_core_model_base_url}")
            return ""
                            # Re-raise so the caller can handle the failure
    
    def _is_symptom_standard(self, question: str) :
        """Return the matched standard symptom for a question, or '' if none matches"""
  
        question_lower = question.lower().strip()

        if question_lower in self.question_to_symptom:
            return self.question_to_symptom[question]
        return ''
    
    def _match_first_question(self,response):
        "Get the first question contained in the response"
        answer=response.split("</think>")[-1]  # Extract the first question
        match = re.search(r'[^?？]*[?？]', answer)
        if match:
            first_question = match.group()
            return first_question.strip()
        return ""
            
    def _update_history_status(self,core_llm_response: str):
        """Update conversation messages, the symptom history queue, and asked-question history"""
        # self.ssdf_core_messages.append({"role": "assistant", "content": core_llm_response})
        
        first_question=self._match_first_question(core_llm_response).strip()
        print(f"===Current first_question content: {first_question}===")

        if first_question:
    
            # Update asked-question tracking
            self.asked_questions.append(first_question)
            if first_question in self.question_to_symptom:
                self.symptom_history.append(self.question_to_symptom[first_question])
                # Enforce the queue length with FIFO semantics
                if len(self.symptom_history) > self.history_length:
                    self.symptom_history.pop(0)
                print(f"===Current symptom_history queue: {self.symptom_history}===")
            else:
                print("===No matching standard symptom!===")
    
    def _collaborative_consultation(self, messages: str):
        """Primary collaborative consultation routine"""
        # print(f"患者回复：{messages[-1]}")

        # First-round question (decided by the LLM)
        if len(messages)<3:
            first_response=self._core_llm(messages)
            # print(f"LLM提问：{first_response}")
            self._update_history_status(first_response)
            # Determine whether it matches the collaboration criteria
            return first_response
        else:
            # Check whether the previous (t-1) question maps to a standard symptom
            if not self._is_symptom_standard(self.asked_questions[-1]):# no standard symptom match
                # If not a standard symptom, let the LLM decide on its own
                print("===Previous question could NOT match a standard symptom; LLM will decide independently===")
                logger.info(f"===Previous question: {self.asked_questions[-1]} could NOT match a standard symptom; LLM generated the next question. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
                current_question=self._core_llm(messages)
            
            else:
                current_question=""
                # Otherwise work in collaborative mode
                print("===Previous question matched a standard symptom; entering collaborative decision mode===")
                print(f"===Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")

                # Fetch Navigator action probabilities
                predicted_symptom, topk_symptoms, topk_probabilities, symptom_prob_dict = self._navigator_predict(self.symptom_history, topk=self.navigator_topk)

                if not symptom_prob_dict:
                    print("===Navigator API call failed; falling back to LLM decision===")
                    current_question = self._core_llm(messages)
                else:
                    print(f"===Navigator predicted symptom: '{predicted_symptom}'===")
                    print(f"===Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
                    
                    # Let the LLM guess q(t)
                    retry_count = 0
                    guesses_log = []  # Track every guess and outcome
                    while retry_count <self.max_retry: # up to the configured maximum number of guesses
                        llm_guess = self._core_llm(messages)
                        llm_guess_oure_question=llm_guess.split('</think>')[-1].strip() if '</think>' in llm_guess else llm_guess
                        print(f"===LLM guess attempt {retry_count+1}: {llm_guess_oure_question}===")

                        # Determine whether it maps to a core symptom
                        first_question=self._match_first_question(llm_guess).strip()
                        standard_symptom=self._is_symptom_standard(first_question)# determine the matched standard symptom
                        if standard_symptom: # if a standard symptom is found
                            
                            # Inspect that symptom's probability within p(t)
                            if standard_symptom in symptom_prob_dict:
                                probability = symptom_prob_dict[standard_symptom]
                                print(f"===Matched symptom: {standard_symptom}, probability: {probability:.4f}===")

                                if probability >= self.low_threshold:
                                    # Accept the LLM guess
                                    current_question = llm_guess
                                    guesses_log.append((retry_count+1, llm_guess, standard_symptom, probability, "Success"))
                                    print(f"===Probability exceeds threshold {self.low_threshold}; accept LLM guess (attempt {retry_count+1})===")
                                    break
                                else:
                                    print(f"===Probability below threshold {self.low_threshold}; retry===")
                                    guesses_log.append((retry_count+1, llm_guess_oure_question, standard_symptom, probability, "Low probability"))
                            else:
                                print(f"===Matched symptom not in Navigator distribution; retry===")
                                guesses_log.append((retry_count+1, llm_guess_oure_question, standard_symptom, None, "Not in distribution"))
                        else:
                            print("===LLM guess did not map to a standard symptom; retry===")
                            guesses_log.append((retry_count+1, llm_guess_oure_question, None, None, "Non-standard symptom"))

                        retry_count += 1
                        
                        
                        # Optionally re-select if the question repeats
                        # if current_question in self.asked_questions:
                        #     print("===当前问题已问过，重新选择Navigator推荐的问题===")
                        #     if symptom_prob_dict:
                        #         # Choose an unseen question from the probability distribution
                        #         for symptom, prob in sorted(symptom_prob_dict.items(), key=lambda x: x[1], reverse=True):
                        #             if prob >= self.final_low_threshold:
                        #                 current_question = self.symptom_to_question[symptom]
                        #                 if current_question not in self.asked_questions:
                        #                     break
                        #                 else:
                        #                     current_question = ""  # Reset default when nothing fits
                        #     else:
                        #         current_question = ""  # Reset default when nothing fits

                    # Print the guess summary
                    print(f"\n===LLM guess summary (total {len(guesses_log)} attempts): {guesses_log}===")


                    if retry_count == self.max_retry and not current_question:
                        # After max retries, fall back to the highest Navigator probability
                        print(f"===LLM guesses failed {self.max_retry} times; using Navigator recommendation with probability above {self.final_low_threshold}===")
                        if symptom_prob_dict:
                            max_prob_symptom = max(symptom_prob_dict, key=symptom_prob_dict.get)
                            
                            # Use the symptom-to-question mapping
                            current_only_question = self.symptom_to_question[max_prob_symptom]# TODO: contains only the question text; add CoT display if required
                            
                            current_question=current_only_question
                        else:
                            # If Navigator returned no probabilities, let the LLM decide
                            print("===Navigator probability distribution empty; falling back to LLM===")
                            current_question = llm_guess

            
            self._update_history_status(current_question)
            # Ask the question and gather the answer
            # print(f"LLM提问：{current_question}")
        
        return current_question
    def _load_prompt_template(self,file_name: str) -> str:
        """Load the system template from the shared prompt file."""
        template_path = Path(__file__).resolve().parent / "prompts" / file_name
        try:
            return template_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error(f"Failed to load system template from {template_path}: {exc}")
    def _collaborative_consultation_v2(self, messages: str):
        """Collaborative consultation routine v2.
        Combines the large-model output with Navigator top-k predictions and lets the model rank/select the final question (using the modified dialogue workflow)."""
        # 第一轮问题（由LLM决定）
        if len(messages)<3:
            first_response=self._core_llm(messages)
            # print(f"LLM提问：{first_response}")
            self._update_history_status(first_response)
            #判断是否是
            return first_response
        else:
            # 检查上一轮问题（第t-1轮）是否能对应标准症状
            if not self._is_symptom_standard(self.asked_questions[-1]):#对应不上
                # 不能对应标准症状：LLM自己决策
                print("===Previous question could NOT match a standard symptom; LLM will decide independently===")
                current_question=self._core_llm(messages)
            
            else:
                current_question=""
                # 能对应标准症状：协同决策

                print(f"===Previous question matched a standard symptom; entering collaborative decision mode. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
                logger.info(f"===Previous question matched a standard symptom; entering collaborative decision mode. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
            
                # 获取Navigator的动作概率分布
                predicted_symptom, topk_symptoms, topk_probabilities, symptom_prob_dict = self._navigator_predict(self.symptom_history, topk=self.navigator_topk)
                print(f"======Navigator predicted symptom: {predicted_symptom}, Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
                logger.info(f"======Navigator predicted symptom: {predicted_symptom}, Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
                #获取对应topk症状的问题symptom_to_question
                topk_questions=[self.symptom_to_question[symptom] for symptom in topk_symptoms]
                
                # LLM直接输出问题和cot
                current_cot_question = self._core_llm(messages)
                #获取单纯问题
                current_question=current_cot_question.split("</think>")[-1].strip() if "</think>" in current_cot_question else current_cot_question

                topK_1_question=topk_questions+[current_question]
                
                #大模型选择一个问题进行返回
                sys_prompt=self._load_prompt_template("consultation_system_prompt.txt")

                prompt_tmp=self._load_prompt_template("consultation_prompt.txt")
                try:
                    dialog=self.format_dialogue(self.ssdf_core_messages)
                    prompt=prompt_tmp.format(messages=dialog, topK_1_question=topK_1_question)
                    # print(prompt)
                    resp=self.llm_generation_and_check(prompt, type=str, system=sys_prompt)
                    
                    resp_json=self.clean_response_to_json(resp)
                    ranked_questions = resp_json.get("result", [])
                except Exception as e:
                    logger.warning(f"LLM failed to rank questions: {e}, model output: {resp}")
                    ranked_questions = ["NULL"]
                
                selected_question = ""
                for candidate_question in ranked_questions:
                    candidate_symptom = self.question_to_symptom.get(candidate_question, "")
                    if candidate_symptom and candidate_symptom in self.symptom_history:
                        continue
                    selected_question = candidate_question
                    break

                top_1_question = selected_question or (ranked_questions[0] if ranked_questions else current_question)
                print(f"===LLM ranked Navigator Top-K questions plus its own question; sorted result {resp_json}, final choice: {top_1_question}===")
                logger.info(f"===LLM ranked Navigator Top-K questions plus its own question; sorted result {resp_json}, final choice: {top_1_question}===")
                
                if top_1_question == current_question:
                    current_question = current_cot_question
                else:
                    current_question = top_1_question
            
        self._update_history_status(current_question)
        # 提问并获取回答
        # print(f"LLM提问：{current_question}")
        
        return current_question
    
    def _collaborative_consultation_v3(self, messages: str):
        """Collaborative consultation routine v3.
        Uses the large-model output together with Navigator top-k suggestions and allows the model to pick the final question (following the adjusted workflow)."""
        # 第一轮问题（由LLM决定）
        if len(messages)<3:
            first_response=self._core_llm(messages)
            # print(f"LLM提问：{first_response}")
            self._update_history_status(first_response)
            #判断是否是
            return first_response
        else:
            # 检查上一轮问题（第t-1轮）是否能对应标准症状
            if not self._is_symptom_standard(self.asked_questions[-1]):#对应不上
                # 不能对应标准症状：LLM自己决策
                print("===Previous question could NOT match a standard symptom; LLM will decide independently===")
                logger.info(f"===Previous question: {self.asked_questions[-1]} could NOT match a standard symptom; LLM generated the next question. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
                current_question=self._core_llm(messages)
            else:
                current_question=""
                # 能对应标准症状：协同决策

                print(f"===Previous question matched a standard symptom; entering collaborative decision mode. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
                logger.info(f"===Previous question matched a standard symptom; entering collaborative decision mode. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
                # 获取Navigator的动作概率分布
                predicted_symptom, topk_symptoms, topk_probabilities, symptom_prob_dict = self._navigator_predict(self.symptom_history, topk=self.navigator_topk)
                print(f"======Navigator predicted symptom: {predicted_symptom}, Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
                logger.info(f"======Navigator predicted symptom: {predicted_symptom}, Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
                # LLM直接输出问题和cot
                count=0
                while True:
                    try:
                        current_cot_question = self._core_llm(messages)
                    except Exception as e:
                        logger.warning(f"LLM failed to generate a question: {e}, model input: {messages}")
                    ss=self.question_to_symptom.get(current_cot_question.split("</think>")[-1].strip() if "</think>" in current_cot_question else current_cot_question, "")
                    if ss in self.symptom_history:
                        logger.info(f"===LLM question: {current_cot_question}, matched standard symptom: {ss}===")
                        count+=1
                        if count==5:
                            self.symptom_history.remove(ss)
                            break
                        continue
                    else:
                            break
                #获取单纯问题
                current_question=current_cot_question.split("</think>")[-1].strip() if "</think>" in current_cot_question else current_cot_question

                #判断如果#获取top1 probability>0.8的症状
                top1_prob=topk_probabilities[0] if topk_probabilities else 0
                if top1_prob>0.9 and  predicted_symptom not in self.symptom_history:
                    current_question=self.symptom_to_question[predicted_symptom]
                    print(f"===Top-1 probability > 0.9; using Navigator-recommended question: {current_question}===")
                    logger.info(f"===Top-1 probability > 0.9; using Navigator-recommended question: {current_question}===")
                else:
                    current_question=current_cot_question
                    print(f"===Top-1 probability <= 0.9; using LLM-generated question: {current_cot_question}===")
                    logger.info(f"===Top-1 probability <= 0.9; using LLM-generated question: {current_cot_question}===")
            
        self._update_history_status(current_question)
        # 提问并获取回答
        # print(f"LLM提问：{current_question}")
        
        return current_question
    
    # def _collaborative_consultation_v4(self, messages: str):
    #     """Collaborative consultation routine v4.
    #     Feeds the large-model output plus Navigator top-k suggestions (with additional history context) into the model to select the final question via the modified workflow."""
    #     # 第一轮问题（由LLM决定）
    #     if len(messages)<3:
    #         first_response=self._core_llm(messages)
    #         # print(f"LLM提问：{first_response}")
    #         self._update_history_status(first_response)
    #         #判断是否是
    #         return first_response
    #     else:
    #         # 检查上一轮问题（第t-1轮）是否能对应标准症状
    #         if not self._is_symptom_standard(self.asked_questions[-1]):#对应不上
    #             # 不能对应标准症状：LLM自己决策
    #             print("===Previous question could NOT match a standard symptom; LLM will decide independently===")
    #             current_question=self._core_llm(messages)
            
    #         else:
    #             current_question=""
    #             # 能对应标准症状：协同决策

    #             print(f"===Previous question matched a standard symptom; entering collaborative decision mode. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
    #             logger.info(f"===Previous question matched a standard symptom; entering collaborative decision mode. Symptom history queue (length={len(self.symptom_history)}): {self.symptom_history}===")
            
    #             # 获取Navigator的动作概率分布
    #             predicted_symptom, topk_symptoms, topk_probabilities, symptom_prob_dict = self._navigator_predict(self.symptom_history, topk=self.navigator_topk)
    #             print(f"======Navigator predicted symptom: {predicted_symptom}, Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
    #             logger.info(f"======Navigator predicted symptom: {predicted_symptom}, Navigator Top-{self.navigator_topk} probability distribution: {symptom_prob_dict}===")
    #             #获取对应topk症状的问题symptom_to_question
    #             topk_questions=[self.symptom_to_question[symptom] for symptom in topk_symptoms]
                
    #             # LLM直接输出问题和cot
    #             current_cot_question = self._core_llm(messages)
    #             #获取单纯问题
    #             current_question=current_cot_question.split("</think>")[-1].strip() if "</think>" in current_cot_question else current_cot_question

    #             topK_1_question=topk_questions+[current_question]
                
    #             #大模型选择一个问题进行返回
    #             sys_prompt="""你是一名中西医结合诊断专家，擅长处理脾胃病领域的疾病诊治。
    # 你遵循“从主到次、从现在到既往、从症状到诱因”的递进逻辑进行问诊。    
    # 具体规则如下：
    # 1. 主诉优先，聚焦核心症状规律：首先明确患者最痛苦的核心症状（如“胃痛3个月，加重1周”“反酸、烧心1个月”），围绕核心症状展开细节追问，包括：（1）症状性质（如胃痛为刺痛、胀痛、隐痛、灼痛，反酸是否伴胸骨后烧灼感）；（2）发作时间（空腹、餐后、夜间、周期性）；（3）持续时长与缓解方式（如休息后缓解、进食温食缓解、服用抑酸药缓解）；（4）症状程度（轻度、中度、重度，是否影响生活）。
    # 2. 由今及昔，追溯病史演变规律：在明确现病史的基础上，逐步追溯既往史（如既往胃炎、胃溃疡、胆囊炎病史）、手术史（如胃肠手术史）、用药史（如长期服用非甾体抗炎药、激素史，中药服用史）；中医需额外关注既往体质（如是否长期怕冷、易疲劳）、外感病史（如近期是否感冒）对脾胃功能的影响。
    # 3. 关联诱因，兼顾内外因素规律：脾胃病发作多与内外因相关，需全面追问：（1）饮食诱因（如暴饮暴食、辛辣刺激、生冷饮食、不洁饮食、饮酒、浓茶咖啡摄入）；（2）情志诱因（如近期压力大、焦虑、生气、熬夜）；（3） 环境与劳累诱因（如受凉、过度劳累）；（4）其他诱因（如服用特定药物后发作）。
    # 4. 系统关联，排查合并症状规律：脾胃与其他脏腑关联密切，需排查相关系统伴随症状：（1）消化系统关联症状（如恶心呕吐、腹胀、嗳气、食欲不振、便秘、腹泻、黑便、便血、黄疸）；（2）全身伴随症状（如体重下降、乏力、贫血、发热）；（3）中医特殊伴随症状（如口干口苦、口淡无味、口臭、喜食热饮/冷饮、大便黏滞、小便黄赤/清长）。"""

    #             prompt_tmp="""/no_think\n请按照脾胃病诊断的问诊规则,参考【已问诊的记录】，对如下【候选问诊问题】中多个记录进行排序。
    # 【候选问诊问题】：{topK_1_question}
    # 要求：
    # 1. 【候选问诊问题】的排序应根据具体诊断规则，使得问诊流程更符合临床逻辑。
    # 2. 不要增加、删除、修改【候选问诊问题】中的内容，只进行排序。
    # 3. 输出以下JSON格式，格式如下：
    # {{"result":["问题1"，"问题2"，...]}}
    # """         
    #             try:
    #                 # dialog=self.format_dialogue(self.ssdf_core_messages)
    #                 prompt=prompt_tmp.format(topK_1_question=topK_1_question)
    #                 # print(prompt)
    #                 resp=self.llm_generation_and_check(prompt, type=str, system=sys_prompt)
                    
    #                 resp_json=self.clean_response_to_json(resp)
    #                 ranked_questions = resp_json.get("result", [])
    #             except Exception as e:
    #                 logger.warning(f"LLM failed to rank questions: {e}, model output: {resp}")
    #                 ranked_questions = ["NULL"]
                
    #             selected_question = ""
    #             for candidate_question in ranked_questions:
    #                 candidate_symptom = self.question_to_symptom.get(candidate_question, "")
    #                 if candidate_symptom and candidate_symptom in self.symptom_history:
    #                     continue
    #                 selected_question = candidate_question
    #                 break

    #             top_1_question = selected_question or (ranked_questions[0] if ranked_questions else current_question)
    #             print(f"===LLM ranked Navigator Top-K questions plus its own question; sorted result {resp_json}, final choice: {top_1_question}===")
    #             logger.info(f"===LLM ranked Navigator Top-K questions plus its own question; sorted result {resp_json}, final choice: {top_1_question}===")
                
    #             if top_1_question == current_question:
    #                 current_question = current_cot_question
    #             else:
    #                 current_question = top_1_question
            
    #     self._update_history_status(current_question)
    #     # 提问并获取回答
    #     # print(f"LLM提问：{current_question}")
        
    #     return current_question
    
    def clean_response_to_json(self,text: str):
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

    
    def llm(self,prompt, system="You are a helpful assistant."):

        messages = []
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.ssdf_core_client.chat.completions.create(
        model=self.ssdf_core_model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        stream=False,
        # 不再传 tools，避免误触发工具导致响应结构变化
        ) 
        return response.choices[0].message.content
    
    def  llm_generation_and_check(self,prompt,type=str, system="You are a helpful assistant."):
        resp_str = None
        retry_times = 0
        max_retries = 5

        while not isinstance(resp_str, type) and retry_times < max_retries:
            try:
                resp_str = self.llm(prompt,system)
                
            except Exception as e:
                retry_times += 1
                print(f"llm_judge generation error.Retry {retry_times}/{max_retries} times, due to error: {e}")

        # 如果重试多次仍然失败，返回空字符串，避免后续 split 报错
        if not isinstance(resp_str, type):
            return ""
        return resp_str

    def format_dialogue(self,messages):
        """
        将 OpenAI-style 对话格式整理为：
        医生：
        患者：
        的交替文本
        """

        role_map = {
            "user": "患者",
            "assistant": "医生"
        }

        lines = []

        for msg in messages:
            role = role_map.get(msg.get("role"), "未知")
            content = msg.get("content", "").strip()

            lines.append(f"{role}：{content}")

        return "\n".join(lines)

if __name__ == "__main__":
    pass
