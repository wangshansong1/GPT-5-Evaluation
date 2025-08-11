class PromptTemplates:
    def __init__(self):
        pass

    def load_templates(self, dataset, model):
        # System Role
        self.zero_shot_system_role = "You are a helpful assistant."
        self.zero_shot_ao_system_role = "Please answer only."
        self.few_shot_system_role = "Follow the given examples and answer the question."

        # Few-shot Trigger
        self.few_shot_trigger = "The answer is"

        # Few-shot Prompt
        self.few_shot_prompt = "Q: {question}\nA:"

        if dataset in ["medqa", "medmcqa", "mmlu_medical"]:
            self.zero_shot_system_role = "You are a helpful medical assistant."

            self.zero_shot_ao_trigger = "Among A through D, the answer is"
            self.zero_shot_cot_trigger = "Therefore, among A through D, the answer is"
        elif dataset in ["medxpertqa", "medxpertqa_sampled", "usmlestep1", "usmlestep2", "usmlestep3"]:
            self.zero_shot_system_role = "You are a helpful medical assistant."

            self.zero_shot_ao_trigger = "Among {start} through {end}, the answer is"
            self.zero_shot_cot_trigger = "Therefore, among {start} through {end}, the answer is"
        elif dataset in ["vqarad"]:
            self.zero_shot_system_role = "You are a helpful medical assistant."

            self.zero_shot_ao_trigger = "Among A through B, the answer is"
            self.zero_shot_cot_trigger = "Therefore, among A through B, the answer is"
        elif dataset in ["medqamainland"]:
            self.zero_shot_system_role = "您是一位乐于助人的医疗助理。"

            self.zero_shot_ao_trigger = "从 A 到 E, 答案是"
            self.zero_shot_cot_trigger = "因此, 在 A 到 E 中, 答案是"
        elif dataset in ["medqataiwan"]:
            self.zero_shot_system_role = "您是一位樂於助人的醫療助理。"

            self.zero_shot_ao_trigger = "从 A 到 D, 答案是"
            self.zero_shot_cot_trigger = "因此, 在 A 到 D 中, 答案是"
        else:
            raise ValueError("Dataset prompt template is not defined...")

        if model == "deepseek-reasoner":
            self.zero_shot_cot_trigger = "Put your final answer within \\boxed{{}}. " + self.zero_shot_cot_trigger
        elif "qvq" in model.lower():
            self.zero_shot_ao_system_role = "You are a helpful medical assistant."
            self.zero_shot_ao_trigger = "Let's think step by step."

        self.zero_shot_ao_prompt = "Q: {question}\nA: " + self.zero_shot_ao_trigger
        self.zero_shot_cot_prompt = "Q: {question}\nA: Let's think step by step."

        return self
