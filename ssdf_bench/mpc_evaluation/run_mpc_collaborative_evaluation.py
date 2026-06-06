from kg_dmcra.coordination.llm_navigator_coordination import SSDFConsultationClient
from ssdf_bench.mpc_evaluation.llm_patient_simulator import PatientSimulator
from openai import OpenAI
import logging
import os
from datetime import datetime
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== Logging Configuration ====================
def setup_logger(fusion_method: str, log_dir: str = "logs") -> logging.Logger:
    """Configure the root logger so every module shares the same output"""

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


# DEFAULT_SYSTEM_PROMPT = """你是一名中西医结合诊断专家，擅长处理脾胃病领域的疾病诊治。\n请遵循如下规则对患者进行系统化问诊，收集完整的病史信息以辅助诊断。\n问诊规则：\n1.请遵循"从主到次、从现在到既往、从症状到诱因"的递进逻辑进行问诊。\n    (1)主诉优先，聚焦核心症状规律：首先明确患者最痛苦的核心症状，例如胃痛3个月，加重1周，反酸、烧心1个月...等，然后围绕核心症状展开细节追问，包括发作时间（如空腹、餐后、夜间、周期性），症状性质（如胃痛为刺痛、胀痛、隐痛、灼痛，反酸是否伴胸骨后烧灼感），持续时长与缓解方式（如休息后缓解、进食温食缓解、服用抑酸药缓解），症状程度（如轻度、中度、重度，是否影响生活）。\n    (2)由今及昔，追溯病史演变规律:在明确现病史的基础上，逐步追溯既往史（如既往胃炎、胃溃疡、胆囊炎病史）、手术史（如胃肠手术史）、用药史（如长期服用非甾体抗炎药、激素史，中药服用史）；中医需额外关注既往体质（如是否长期怕冷、易疲劳）、外感病史（如近期是否感冒）对脾胃功能的影响。\n    (3)关联诱因，兼顾内外因素规律:全面追问脾胃病发作的相关内外因素，包括饮食诱因（如暴饮暴食、辛辣刺激、生冷饮食、不洁饮食、饮酒、浓茶咖啡摄入）、情志诱因（如近期压力大、焦虑、生气、熬夜）、环境与劳累诱因（如受凉、过度劳累）、其他诱因（如服用特定药物后发作）。\n    (4)系统关联，排查合并症状规律：脾胃与其他脏腑关联密切，需排查相关系统伴随症状，例如消化系统关联症状（如恶心呕吐、腹胀、嗳气、食欲不振、便秘、腹泻、黑便、便血、黄疸）、全身伴随症状（如体重下降、乏力、贫血、发热）、中医特殊伴随症状（如口干口苦、口淡无味、口臭、喜食热饮/冷饮、大便黏滞、小便黄赤/清长）。\n2.当未收集到足够的症状信息时，需要进行单词问诊，注意每次只能询问一个问题，且问诊内容与之前问过的内容需保持逻辑连贯与层次递进，避免重复问诊及跨阶段跳跃式提问，以保障信息收集的全面性、系统性和可用性。\n3.当收集到足够多患者症状信息后询问"请问您是否还有其他不适症状？"\n4.进行总结陈述，内容包括：\n    (1)症状总结：简要概括患者的主要症状、持续时间及相关特点。\n    (2)诊断结果：基于收集的病史信息，给出中医证型和西医诊断。"""
DEFAULT_SYSTEM_PROMPT = """你是一名中西医结合诊断专家，擅长处理脾胃病领域的疾病诊治。\n请遵循如下规则对患者进行系统化问诊，需与患者进行多次交互问诊，收集完整的病史信息以辅助诊断。\n问诊规则：\n1.每次只能询问一个问题，包含一个患者可能存在的症状，如“您是否存在胀气”，且直接返回问诊问题，不得一次询问多个症状内容，如“请问您是否有食欲减退、疲倦乏力、舌苔厚腻或脉象濡缓等其他症状？”。\n2.请遵循"从主到次、从现在到既往、从症状到诱因"的递进逻辑进行问诊。\n3.当未收集到足够的症状信息时，需要进行单次问诊，注意每次只能询问一个问题，且问诊内容与之前问过的内容需保持逻辑连贯与层次递进，避免重复问诊及跨阶段跳跃式提问，以保障信息收集的全面性、系统性和可用性。\n4.当收集到足够多患者症状信息后询问"请问您是否还有其他不适症状？"\n5.进行总结陈述，内容包括：\n    (1)症状总结：简要概括患者的主要症状、持续时间及相关特点。\n    (2)诊断结果：基于收集的病史信息，给出中医证型和西医诊断。"""
# DEFAULT_SYSTEM_PROMPT=""

def process_single_item(k: int, item: dict, args: argparse.Namespace):
    logger = logging.getLogger(__name__)

    # Retrieve patient medical record
    medical_record = item["patient_summary"]
    main_symptom = item["chat_rounds"]["messages"][1]["content"]  # The first inquiry entry is usually the chief complaint

    # Create doctor client (SSDF collaborative consultation)
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
        # Direct consultation with the base LLM
        api_key = os.getenv("OPENAI_API_KEY", "-")
        doctor_client = OpenAI(api_key=api_key, base_url=args.ssdf_core_model_base_url)

    # Create patient simulator based on the medical case
    patient = PatientSimulator(medical_record=medical_record, patient_model_name=args.patient_model_name, patient_base_url=args.patient_base_url)
    patient.conversation_history.append({
        "doctor": "",
        "patient": main_symptom
    })

    # Initialize messages with the chief complaint from the case record
    messages = [
        {"role": "system", "content": args.ssdf_core_sys_prompt},
        {"role": "user", "content": f"{main_symptom}"}
    ]
    messages_cot=[
        {"role": "system", "content": args.ssdf_core_sys_prompt},
        {"role": "user", "content": f"{main_symptom}"}
    ]

    logger.info(f"[Progress] Start processing sample {k+1}")
    logger.debug(f"[Chief Complaint]: {main_symptom}")
    # Automatic consultation loop
    patient_end_judge = "no"
    for round_num in range(30):
        logger.info(f"======================== Consultation round {round_num+1} =================================")
        # Doctor questions (collaborative or standalone)
        if args.use_ssdf_navigator:
            doctor_cot_question = doctor_client._collaborative_consultation_v2(messages)
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
                logger.error(f"Doctor model invocation failed: {e}")
                doctor_cot_question=""

        doctor_question = doctor_cot_question.split("</think>")[-1].strip() if "</think>" in doctor_cot_question else doctor_cot_question
        logger.info(f"[Doctor]: {doctor_cot_question}")

        # Check whether conversation should end
        patient_end_judge = patient.judge_end(doctor_question.split("</think>")[-1].strip() if "</think>" in doctor_question else doctor_question)
        if "yes" in patient_end_judge:
            messages.append({"role": "assistant", "content": doctor_question})  # Store doctor's conclusion
            messages_cot.append({"role": "assistant", "content": doctor_cot_question})  # Store doctor's conclusion
            break

        # Patient responds according to the medical record
        patient_answer = patient.respond(doctor_question)
        logger.info(f"[Patient]: {patient_answer}")

        # Update conversation history
        messages.append({"role": "assistant", "content": doctor_question})
        messages.append({"role": "user", "content": patient_answer})

        messages_cot.append({"role": "assistant", "content": doctor_cot_question})
        messages_cot.append({"role": "user", "content": patient_answer})

    # Force conclusion generation when needed
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
                logger.error(f"Doctor model invocation failed: {e}")
                doctor_question=""

            doctor_question = doctor_cot_question.split("</think>")[-1].strip() if "</think>" in doctor_cot_question else doctor_cot_question
        logger.info(f"Diagnosis conclusion: {doctor_question}")

        # At max rounds, force the model to summarize
        messages.append({"role": "assistant", "content": doctor_question})
        messages_cot.append({"role": "assistant", "content": doctor_cot_question})

    # Record completion info
    logger.info(f"Sample {k+1} completed")

    item["exam_chat_rounds"] = messages
    item["exam_chat_rounds_cot"] = messages_cot
    return k, item

def main():
    parse = argparse.ArgumentParser()
    # Basic arguments
    parse.add_argument("--fusion_method", default="Qwen3-14B", help="How navigator and LLM converge.")

    parse.add_argument("--file-diagnose-records", default="./data/exam_datas.json",
                      help="Path to evaluation consultation records (JSON).")
    # parse.add_argument("--base-url", default="http://localhost:8009/v1", help="Base URL of the LLM under evaluation.")
    # parse.add_argument("--model", default="ssdf-consultation", help="Name of the model under evaluation.")
    parse.add_argument("--file-save-path", default="./results/consultation_results.json",
                      help="Path for saving consultation records.")
    ## LLM and Navigator collaboration parameters
    parse.add_argument("--ssdf-core-model", default="test", help="SSDF-core model name.")
    parse.add_argument("--ssdf-core-model-base-url", default="http://localhost:8000/v1",
                      help="SSDF-core model Base URL.")
    parse.add_argument("--ssdf_core_sys_prompt", default=DEFAULT_SYSTEM_PROMPT,
                      help="SSDF-core model system prompt and model to be tested (doctor model) system prompt.")
    parse.add_argument("--history-length", type=int, default=5, help="Symptom history cohort length.")
    parse.add_argument("--low-threshold", type=float, default=0.1,
                      help="Navigator probability minimum threshold.")
    parse.add_argument("--final-low-threshold", type=float, default=0.9,
                      help="When the LLM guesses three times are not in the navigator, the final probability threshold of the navigator is the lowest.")
    parse.add_argument("--max-retry", type=int, default=1, help="Maximum number of retries in SSDF-core.")
    parse.add_argument("--navigator-url", default="http://localhost:8999/predict",
                      help="Navigator service URL.")
    parse.add_argument("--navigator-topk", type=int, default=5,
                      help="Navigator returns the number of Top-K symptoms.")
    parse.add_argument("--use-ssdf-navigator",
                       action="store_false",
                       default=False,
                       help="Disable SSDF-Navigator collaboration")
    # Patient simulator arguments
    parse.add_argument("--patient-model-name", default="Qwen3-32B",
                      help="Model name used for the patient simulator.")
    parse.add_argument("--patient-base-url", default="http://localhost:8000/v1",
                      help="Base URL of the patient simulator model service.")

    parse.add_argument("--symptom_file",
                      default="./data/symptoms.csv",
                      help="Standard symptom file path.")
    parse.add_argument("--question_cleaned_file",
                      default="./data/question_cleaned.csv",
                      help="Cleaned question file path.")
    args = parse.parse_args()
    
    # Initialize logger
    logger = setup_logger(args.fusion_method)
    logger.info(f"=" * 60)
    logger.info(f"Evaluation started - fusion method: {args.fusion_method}")
    logger.info(f"Input evaluation file: {args.file_diagnose_records}")
    logger.info(f"Output save path: {args.file_save_path}")
    logger.info("Runtime arguments:\n%s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    logger.info(f"=" * 60)

    # Read test consultation records
    with open(args.file_diagnose_records, "r") as f:
        json_data = json.load(f)
    
    # Store model response records
    messages_exam_records = []
    ordered_results_buffer = {}
    next_to_save = 0

    # Run evaluation in 8 processes while keeping output order stable
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_index = {
            executor.submit(process_single_item, k, item, args): k
            for k, item in enumerate(json_data)
        }

        for future in as_completed(future_to_index):
            k, processed_item = future.result()
            ordered_results_buffer[k] = processed_item

            # Save each completed item once, but only when all prior items are ready
            while next_to_save in ordered_results_buffer:
                messages_exam_records.append(ordered_results_buffer.pop(next_to_save))
                with open(args.file_save_path, "w") as f:
                    json.dump(messages_exam_records, f, ensure_ascii=False, indent=4)
                next_to_save += 1

    logger.info(f"=" * 60)
    logger.info(f"All samples evaluated! Total processed: {len(json_data)}")
    logger.info(f"=" * 60)

if __name__ == "__main__":
    main()
