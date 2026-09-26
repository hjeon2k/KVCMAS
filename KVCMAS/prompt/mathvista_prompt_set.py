"""MathVista prompt set. """
from KVCMAS.prompt.gsm8k_prompt_set import GSM8KPromptSet
from KVCMAS.prompt.prompt_set_registry import PromptSetRegistry


@PromptSetRegistry.register('mathvista')
class MathVistaPromptSet(GSM8KPromptSet):
    @staticmethod
    def get_constraint(role=None):
        # MathVista answers carry units and choice text, and the inherited GSM8K text
        # also promises few-shot examples that never come.
        try:
            base = GSM8KPromptSet.get_constraint(role)
        except TypeError:
            base = GSM8KPromptSet.get_constraint()
        base = str(base).replace(
            "The last line of your output contains only the final result without any units, "
            "for example: The answer is 140",
            "End with the final answer on its own last line, keeping units or the choice "
            "letter when applicable, for example: 'the answer is 140' or 'the answer is (B)'"
        ).replace("You will be given some examples you may refer to.", "")
        return base

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a visual math problem, its analysis and results from other agents. "
            "Find the most reliable answer based on their analysis. Give reasons for the decision. "
            "For multiple-choice questions answer with the letter of the correct choice. "
            "End with the final answer on its own last line, "
            "for example: 'the answer is 140' or 'the answer is (B)'")
