import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI


# ==================== Shared Logger ====================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PatientSimulator:
    """Patient simulator used to generate responses aligned with the patient's condition"""

    def __init__(self, patient_base_url="http://localhost:11117/v1",patient_model_name="Qwen3-32B",chief_complaint: str="", additional_info: str = "",
                 system_template: Optional[str] = None, medical_record: Optional[str] = None):
        """
        Initialize patient simulator
        """
        self.patient_model_name = patient_model_name
        self.patient_base_url = patient_base_url
        # NOTE: Some local OpenAI-compatible backends ignore api_key, so a
        # placeholder is accepted. For cloud APIs, set OPENAI_API_KEY env var.
        api_key = os.environ.get("OPENAI_API_KEY", "-")
        self.client = OpenAI(api_key=api_key, base_url=patient_base_url)
        
        logger.info(f"Patient simulator initialized - Model: {patient_model_name}, API: {patient_base_url}")
        
        self.medical_record = medical_record
        if medical_record:
            self.chief_complaint = "基于医案记录"
            self.additional_info = "基于医案记录"
            self.system_template = self._load_system_template()
            logger.debug("Using medical record mode")
        else:
            logger.debug(f"Using default mode - Chief complaint: {chief_complaint}")

        self.conversation_history: List[Dict[str, str]] = []
        self._update_system_prompt()

        print("==============================")
        print(self.system_template)
    def _load_system_template(self) -> str:
        """Load the system template from the shared prompt file."""
        template_path = Path(__file__).resolve().parent / "prompts" / "medical_recoard_system_prompt.txt"
        try:
            return template_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error(f"Failed to load system template from {template_path}: {exc}")
    def _update_system_prompt(self):
        """Update system prompt"""
        if self.medical_record:
            self.system_prompt = self.system_template.format(
                medical_record=self.medical_record
            )
        else:
            self.system_prompt = self.system_template.format(
                chief_complaint=self.chief_complaint,
                additional_info=self.additional_info if self.additional_info else "无特殊说明"
            )

    def _llm_patient(self,prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Generic LLM patient interface"""
        if system is None:
            system = "你是一位患者，请根据你的症状如实回答医生的问题。保持回答简洁，只回答询问的内容。"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('max_tokens', 8192)

        try:
            response = self.client.chat.completions.create(
                model=self.patient_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "我不太清楚这个问题。"

    def build_conversation_history(self):
        """Build conversation history text"""
        history_parts = []
        
        for entry in self.conversation_history:
            patient_text = entry.get('patient', '').strip()
            if not patient_text:
                continue
                
            doctor_text = entry.get('doctor', '').strip() if entry.get('doctor') is not None else ''
            
            if doctor_text:
                history_parts.append(f"医生：{doctor_text}\n患者：{patient_text}")
            else:
                history_parts.append(f"患者：{patient_text}")
        
        return "\n".join(history_parts)

    def respond(self, doctor_question: str, include_history: bool = True) -> str:
        """Generate patient response"""
        logger.debug(f"Generating patient reply - Doctor question: {doctor_question[:50]}...")
        
        if include_history and self.conversation_history:
            history_text = self.build_conversation_history()
            full_prompt = f"""/no_think\n以下是之前的问诊对话：
{history_text}

医生新问题：{doctor_question}

请根据你的症状和之前的回答，继续回答医生的问题："""
        else:
            full_prompt = doctor_question

        response = self._llm_patient("/no_think\n"+full_prompt, self.system_prompt)
        response = response.split("</think>")[-1].strip()
        
        logger.debug(f"Patient reply generated: {response[:50]}...")
        
        self.conversation_history.append({
            "doctor": doctor_question,
            "patient": response
        })

        return response

    def judge_end(self, doctor_question: str) -> bool:
        """Determine whether the consultation should end"""
        logger.debug(f"Evaluating consultation end - Doctor reply: {doctor_question[:50]}...")
        
        prompt=f"/no_think\n请判断下面的医生回复是否包含诊断结论，返回json格式{{'results':yes/no}},yes表示包含诊断结果，no表示未包含诊断结果。\n\n医生回复：{doctor_question}"
        messages=[
            {"role": "system", "content": "You are a helpfull assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
                model=self.patient_model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                stream=False,
            )
        result = response.choices[0].message.content
        logger.debug(f"Consultation end judgment: {result}")
        return result

    def get_conversation_summary(self) -> str:
        """Get conversation summary"""
        summary_parts = []
        for i, entry in enumerate(self.conversation_history, 1):
            doctor_text = entry.get('doctor', '').strip()
            patient_text = entry.get('patient', '').strip()
            if doctor_text:
                summary_parts.append(f"第{i}轮 - 医生: {doctor_text}")
            if patient_text:
                summary_parts.append(f"第{i}轮 - 患者: {patient_text}")
        return "\n".join(summary_parts)


# Demo code
if __name__ == "__main__":
    pass

