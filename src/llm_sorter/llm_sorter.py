"""LLM-powered sorting library using merge-sort with LLM-based comparisons.

This module provides a sorting algorithm that uses Large Language Models
to compare and sort items based on semantic meaning rather than simple
lexicographic or numeric ordering.
"""

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
    """A sorting class that uses LLMs to compare and order items.

    This class implements a merge-sort algorithm where comparisons between
    items are performed by an LLM, enabling semantic sorting based on
    meaning and content rather than simple string or numeric comparison.

    Attributes:
        api_key: The OpenRouter API key for authentication.
        model: The model identifier to use for comparisons. Defaults to "openai/gpt-5-nano".
            For all models supported by OpenRouter, see: https://openrouter.ai/models.
    """

    def __init__(self, api_key: str, model: str = "openai/gpt-5-nano") -> None:
        """Initialize the LLMSorter with API credentials and model selection.

        Args:
            api_key: The OpenRouter API key for authentication.
            model: The model identifier to use for comparisons.
        """
        self.api_key = api_key
        self.model = model

    def sort(
        self,
        *,
        items: list[T],
        prompt: str | None = None,
    ) -> list[T]:
        """Sort a list of items using LLM-based comparisons.

        Implements a merge-sort algorithm where item comparisons are
        performed by an LLM, allowing for semantic sorting based on
        custom criteria specified in the prompt.

        Args:
            items: The list of items to sort. Items can be of any type
                that has a meaningful string representation.
            prompt: Optional custom prompt to guide the LLM's comparison
                logic. If not provided, uses the default comparison prompt.

        Returns:
            A new list containing the items sorted according to the
            LLM's comparison results.
        """
        if len(items) <= 1:
            return items

        mid = len(items) // 2
        left = self.sort(items=items[:mid], prompt=prompt)
        right = self.sort(items=items[mid:], prompt=prompt)

        return self._merge(left=left, right=right, prompt=prompt)


    def _merge(self, *, left: list[T], right: list[T], prompt: str | None) -> list[T]:
        """Merge two sorted lists into a single sorted list.

        Args:
            left: The left sorted sublist.
            right: The right sorted sublist.
            prompt: Optional custom prompt for LLM comparisons.

        Returns:
            A merged list containing all elements from both sublists
            in sorted order.
        """
        merged: list[T] = []
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
        """Compare two items using the LLM to determine their relative order.

        Args:
            first: The first item to compare.
            second: The second item to compare.
            prompt: Optional custom prompt for the comparison logic.
                If not provided, uses the default comparison prompt.

        Returns:
            True if the first item should come before or at the same
            position as the second item, False otherwise.
        """
        comparison_prompt = prompt or DEFAULT_COMPARE_PROMPT
        agent = Agent(
            model=f"openrouter:{self.model}",
            system_prompt=DEFAULT_COMPARE_SYSTEM_PROMPT,
            output_type=bool,
        )
        result = agent.run_sync(f"First value: {first}\nSecond value: {second}\n\n{comparison_prompt}")
        return result.output
