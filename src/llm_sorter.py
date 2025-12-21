from typing import TypeVar

from pydantic_ai import Agent

DEFAULT_COMPARE_SYSTEM_PROMPT = (
    "You are a comparison function for a sorting algorithm. "
    "Your goal is to enable sorting of any objects that have a string representation. "
    "You will be given two values and must determine their relative order. "
    "You must return a boolean: True or False."
)

DEFAULT_COMPARE_PROMPT = (
    "Evaluate each value based on its meaning and content, then determine the sorting order. "
    "Return True if the first value should come before or at the same position as the second value. "
    "Return False if the first value should come after the second value."
)

T = TypeVar("T")

class LLMSorter:
    def __init__(self, api_key: str, model: str = "openai:gpt-5.2"):
        self.api_key = api_key
        self.model = model

    def sort(
        self,
        *,
        items: list[T],
        prompt: str | None = None,
    ) -> list[T]:
        if len(items) <= 1:
            return items

        mid = len(items) // 2
        left = self.sort(items=items[:mid], prompt=prompt)
        right = self.sort(items=items[mid:], prompt=prompt)

        return self._merge(left=left, right=right, prompt=prompt)


    def _merge(self, *, left: list[T], right: list[T], prompt: str | None) -> list[T]:
        merged = []
        i = 0
        j = 0

        while i < len(left) and j < len(right):
            if self._compare(first=left[i], second=right[j], prompt=prompt):
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged


    def _compare(self, *, first: T, second: T, prompt: str | None = None) -> bool:
        comparison_prompt = prompt or DEFAULT_COMPARE_PROMPT
        agent = Agent(
            model=f"gateway/{self.model}",
            system_prompt=DEFAULT_COMPARE_SYSTEM_PROMPT,
            output_type=bool,
        )
        result = agent.run_sync(f"First value: {first}\nSecond value: {second}\n\n{comparison_prompt}")
        return result.output
