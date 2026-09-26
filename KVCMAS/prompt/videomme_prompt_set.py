"""Video-MME prompt set. """
from KVCMAS.prompt.mmlu_prompt_set import MMLUPromptSet
from KVCMAS.prompt.prompt_set_registry import PromptSetRegistry


_REASON = (" First analyze the video evidence and explain your reasoning step by step,"
           " then end with the final line, for example: 'the answer is (B)'")


@PromptSetRegistry.register('videomme')
class VideoMMEPromptSet(MMLUPromptSet):
    @staticmethod
    def get_constraint(role=None):
        try:
            base = MMLUPromptSet.get_constraint(role)
        except TypeError:
            base = MMLUPromptSet.get_constraint()
        return str(base) + _REASON

    @staticmethod
    def get_decision_constraint():
        return (str(MMLUPromptSet.get_decision_constraint())
                + " Weigh the other agents' reasoning, give your own, and end with the"
                  " final line, for example: 'the answer is (B)'")
