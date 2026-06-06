from kg_dmcra.coordination.llm_navigator_coordination import SSDFConsultationClient
from llm_patient_simulator import PatientSimulator
from openai import OpenAI
import logging
import os
from datetime import datetime
import argparse
import json


# ==================== 日志配置 ====================
def setup_logger(fusion_method: str, log_dir: str = "logs") -> logging.Logger:
    """配置根日志器，让所有模块共享同一个日志输出"""

    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_dir)
    os.makedirs(logs_root, exist_ok=True)

    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{fusion_method}_{log_timestamp}.log"
    log_path = os.path.join(logs_root, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)-8s] %(message)s',
        datefmt='%H:%M:%S'
    )

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    quiet_libs = [
    "openai",
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "transformers",
    "torch"
]
    for name in quiet_libs:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """你是一名中西医结合诊断专家，擅长处理脾胃病领域的疾病诊治。\n请遵循如下规则对患者进行系统化问诊，收集完整的病史信息以辅助诊断。\n问诊规则：\n1.请遵循"从主到次、从现在到既往、从症状到诱因"的递进逻辑进行问诊。\n    (1)主诉优先，聚焦核心症状规律：首先明确患者最痛苦的核心症状，例如胃痛3个月，加重1周，反酸、烧心1个月...等，然后围绕核心症状展开细节追问，包括发作时间（如空腹、餐后、夜间、周期性），症状性质（如胃痛为刺痛、胀痛、隐痛、灼痛，反酸是否伴胸骨后烧灼感），持续时长与缓解方式（如休息后缓解、进食温食缓解、服用抑酸药缓解），症状程度（如轻度、中度、重度，是否影响生活）。\n    (2)由今及昔，追溯病史演变规律:在明确现病史的基础上，逐步追溯既往史（如既往胃炎、胃溃疡、胆囊炎病史）、手术史（如胃肠手术史）、用药史（如长期服用非甾体抗炎药、激素史，中药服用史）；中医需额外关注既往体质（如是否长期怕冷、易疲劳）、外感病史（如近期是否感冒）对脾胃功能的影响。\n    (3)关联诱因，兼顾内外因素规律:全面追问脾胃病发作的相关内外因素，包括饮食诱因（如暴饮暴食、辛辣刺激、生冷饮食、不洁饮食、饮酒、浓茶咖啡摄入）、情志诱因（如近期压力大、焦虑、生气、熬夜）、环境与劳累诱因（如受凉、过度劳累）、其他诱因（如服用特定药物后发作）。\n    (4)系统关联，排查合并症状规律：脾胃与其他脏腑关联密切，需排查相关系统伴随症状，例如消化系统关联症状（如恶心呕吐、腹胀、嗳气、食欲不振、便秘、腹泻、黑便、便血、黄疸）、全身伴随症状（如体重下降、乏力、贫血、发热）、中医特殊伴随症状（如口干口苦、口淡无味、口臭、喜食热饮/冷饮、大便黏滞、小便黄赤/清长）。\n2.问诊过程需保持逻辑连贯与层次递进，避免重复问诊及跨阶段跳跃式提问，以保障信息收集的全面性、系统性和可用性。\n3.当收集到足够多患者症状信息后询问"请问您是否还有其他不适症状？"，若患者回答其他症状，则重新进行1.中的问诊；若患者回答无，则进行总结陈述，内容包括：\n    (1)症状总结：简要概括患者的主要症状、持续时间及相关特点。\n    (2)诊断结果：基于收集的病史信息，给出中医证型和西医诊断。"""

    
def main():
    parse = argparse.ArgumentParser()
    #基础参数
    parse.add_argument("--fusion_method", default="PureLLM", help="navigator和LLM的融合方式。")

    parse.add_argument("--file-diagnose-records",
                      default="./data/exam_datas.json",
                      help="Path to evaluation consultation records (JSON).")
    # parse.add_argument("--base-url", default="http://localhost:8009/v1", help="待测LLM服务的Base URL。")
    # parse.add_argument("--model", default="ssdf-consultation", help="待测模型名称。")
    parse.add_argument("--file-save-path",
                      default="./results/consultation_results.json",
                      help="Path for saving consultation records.")
    ##LLM与Navigator协同参数
    parse.add_argument("--ssdf-core-model", default="test", help="SSDF-core模型名称。")
    parse.add_argument("--ssdf-core-model-base-url", default="http://localhost:8000/v1",
                      help="SSDF-core模型Base URL。")
    parse.add_argument("--ssdf_core_sys_prompt", default=DEFAULT_SYSTEM_PROMPT,
                      help="SSDF-core模型系统提示词、待测模型（医生模型）系统提示词。")
    parse.add_argument("--history-length", type=int, default=5, help="症状历史队列长度。")
    parse.add_argument("--low-threshold", type=float, default=0.1, help="Navigator概率最低阈值。")
    parse.add_argument("--final-low-threshold", type=float, default=0.9,
                      help="LLM三次猜测不在navigator中时，Navigator最终概率最低阈值。")
    parse.add_argument("--max-retry", type=int, default=1, help="SSDF-core最大重试次数。")
    parse.add_argument("--navigator-url", default="http://localhost:8999/predict",
                      help="Navigator服务URL。")
    parse.add_argument("--navigator-topk", type=int, default=5, help="Navigator返回Top-K症状数量。")
    parse.add_argument("--use-ssdf-navigator",
                       action="store_false",
                       default=False,
                       help="禁用SSDF-Navigator协同")
    #patient模拟器参数
    parse.add_argument("--patient-model-name", default="Qwen3-32B",
                      help="患者模拟器使用的模型名称。")
    parse.add_argument("--patient-base-url", default="http://localhost:8000/v1",
                      help="患者模拟器使用的模型服务Base URL。")

    parse.add_argument("--symptom_file",
                      default="./data/symptoms.csv",
                      help="标准症状文件路径。")
    parse.add_argument("--question_cleaned_file",
                      default="./data/question_cleaned.csv",
                      help="问题清洗后文件路径。")
    args = parse.parse_args()
    
    # 初始化日志记录器
    logger = setup_logger(args.fusion_method)
    logger.info(f"=" * 60)
    logger.info(f"评估任务开始 - 融合方法: {args.fusion_method}")
    logger.info(f"测评文件: {args.file_diagnose_records}")
    logger.info(f"保存路径: {args.file_save_path}")
    logger.info(f"=" * 60)

    #读取测试问诊记录
    with open(args.file_diagnose_records, "r") as f:
        json_data = json.load(f)

    #测试模型回答记录字典
    messages_exam_records = []
    #进行测评
    for k,item in enumerate(json_data[74:]):
        
        # 获取患者诊断记录
        medical_record = item["patient_summary"]
        main_symptom = item["chat_rounds"]["messages"][1]["content"]  # 第一个问询内容通常是主诉
        
        # 创建医生客户端（SSDF协同问诊）
        if args.use_ssdf_navigator:
                doctor_client = SSDFConsultationClient(
                    ssdf_core_model=args.ssdf_core_model,
                    ssdf_core_model_base_url=args.ssdf_core_model_base_url,
                    ssdf_core_sys_prompt=args.ssdf_core_sys_prompt,
                    history_length=args.history_length,
                    low_threshold=args.low_threshold,
                    final_low_threshold=args.final_low_threshold,
                    max_retry=args.max_retry,
                    navigator_url=args.navigator_url,
                    navigator_topk=args.navigator_topk,
                    symptom_file=args.symptom_file,
                    question_cleaned_file=args.question_cleaned_file
                )
        else:
            #大模型直接问诊
            api_key = os.getenv("OPENAI_API_KEY", "-")
            doctor_client = OpenAI(api_key=api_key, base_url=args.ssdf_core_model_base_url)

        
        # 创建基于医案记录的患者模拟器
        patient = PatientSimulator(medical_record=medical_record, patient_model_name=args.patient_model_name, patient_base_url=args.patient_base_url)
        patient.conversation_history.append({
            "doctor": "",
            "patient": main_symptom
        })
        
        # 初始化消息（使用医案记录中的主诉）
        messages = [
            {"role": "system", "content": args.ssdf_core_sys_prompt},
            {"role": "user", "content": f"{main_symptom}"}
        ]
        messages_cot=[
            {"role": "system", "content": args.ssdf_core_sys_prompt},
            {"role": "user", "content": f"{main_symptom}"}
        ]
        
        logger.info(f"【处理进度】开始处理第 {k+1} 个样本")
        logger.debug(f"[患者主诉]: {main_symptom}")
        # 自动问诊循环
        patient_end_judge = "no"
        for round_num in range(30):
            logger.info(f"======================== 第 {round_num+1} 轮问诊 =================================")
            # 医生提问（可选协同SSDF与非协同决策）
            if args.use_ssdf_navigator:
                doctor_cot_question = doctor_client._collaborative_consultation_v4(messages)
            else:
                try:
                    response = doctor_client.chat.completions.create(
                        # model="piweibing",
                        model=args.ssdf_core_model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=8192
                    )

                    doctor_cot_question = response.choices[0].message.content
                
                except Exception as e:
                    logger.error(f"医生模型调用失败: {e}")
                    doctor_cot_question=""
            doctor_question = doctor_cot_question.split("</think>")[-1].strip() if "</think>" in doctor_cot_question else doctor_cot_question
            logger.info(f"[医生]：{doctor_cot_question}")

            # 检查是否结束
            
            patient_end_judge = patient.judge_end(doctor_question.split("</think>")[-1].strip() if "</think>" in doctor_question else doctor_question)
            if "yes" in patient_end_judge:
                messages.append({"role": "assistant", "content": doctor_question})  # 保存医生结论
                messages_cot.append({"role": "assistant", "content": doctor_cot_question})  # 保存医生结论
                break

            # 患者基于医案记录回答
            patient_answer = patient.respond(doctor_question)
            logger.info(f"[患者]：{patient_answer}")

            # 更新消息历史
            messages.append({"role": "assistant", "content": doctor_question})
            messages.append({"role": "user", "content": patient_answer})
            
            messages_cot.append({"role": "assistant", "content": doctor_cot_question})  
            messages_cot.append({"role": "user", "content": patient_answer})
        
        # 主动结束让模型生成结论
        if "yes" not in patient_end_judge and round_num == 29:

            if args.use_ssdf_navigator:
                doctor_question = doctor_client._generate_diagnose_result(messages)
            else:
                try:
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
            
            #### 西医诊断：填写西医诊断结果。""".format(diagnose=diagnose_records)
                
                    response = doctor_client.chat.completions.create(
                    model=args.ssdf_core_model,
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8192
                )
                    
                    doctor_cot_question = response.choices[0].message.content

                except Exception as e:
                    logger.error(f"医生模型调用失败: {e}")
                    doctor_question=""

                doctor_question = doctor_cot_question.split("</think>")[-1].strip() if "</think>" in doctor_cot_question else doctor_cot_question
            logger.info(f"诊断结论：{doctor_question}")
            
            # 达到最大轮次就让模型生成结论
            messages.append({"role": "assistant", "content": doctor_question})
            messages_cot.append({"role": "assistant", "content": doctor_cot_question})
        
        # 记录完成信息
        logger.info(f"样本 {k+1} 处理完成")
        
        item["exam_chat_rounds"] = messages
        item["exam_chat_rounds_cot"] = messages_cot
        messages_exam_records.append(item)
        with open(args.file_save_path, "w") as f:
            json.dump(messages_exam_records, f, ensure_ascii=False, indent=4)

    logger.info(f"=" * 60)
    logger.info(f"所有样本评估完成! 共处理 {len(json_data)} 个样本")
    logger.info(f"=" * 60)

if __name__ == "__main__":
    main()
