import os

import pytest
from dotenv import load_dotenv

from llm_sorter import LLMSorter

load_dotenv()
@pytest.fixture
def api_key() -> str:
    api_key = os.getenv("PYDANTIC_AI_GATEWAY_API_KEY")
    if api_key is None:
        raise ValueError("PYDANTIC_AI_GATEWAY_API_KEY is not set")
    return api_key


@pytest.mark.parametrize("input_list,expected", [
    ([], []),
    ([5], [5]),
    ([3, 1], [1, 3]),
    ([3, 1, 4, 1, 5, 9, 2, 6], [1, 1, 2, 3, 4, 5, 6, 9]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([2, 2, 2, 1, 1], [1, 1, 2, 2, 2]),
])
def test_sort_integers(api_key: str, input_list: list[int], expected: list[int]):
    sorter = LLMSorter(api_key=api_key)
    assert sorter.sort(items=input_list) == expected


@pytest.mark.parametrize("input_list,expected", [
    ([], []),
    (["a", "b", "c"], ["a", "b", "c"]),
    (["c", "b", "a"], ["a", "b", "c"]),
    (["a", "a", "b", "b", "c", "c"], ["a", "a", "b", "b", "c", "c"]),
])
def test_sort_strings(api_key: str, input_list: list[str], expected: list[str]):
    sorter = LLMSorter(api_key=api_key)
    result = sorter.sort(
        items=input_list,
        prompt="Sort the list of strings in ascending, alphebetical order.")
    assert result == expected


def test_sort_literary_passages_by_complexity(api_key: str):
    passages = [
        # Highest reading level
        (
            "The wind’s passage through the trees became a kind of language, "
            "articulate in its insistence, as if the world were attempting to speak "
            "before being interrupted. Standing at the threshold, Maya sensed the "
            "approaching storm not merely as weather, but as a force of "
            "transformation, one that would fracture the evening’s fragile calm and "
            "reassemble it into something unrecognizable."
        ),
        # Lowest reading level
        (
            "The cat sat on the mat. It saw a bug and jumped up. "
            "The bug flew away. The cat looked and then lay down again."
        ),
        # Middle reading level
        (
            "The wind moved through the trees and made the leaves whisper. "
            "Maya stood on the porch and watched the sky darken. "
            "She felt nervous but excited, knowing the storm would change the quiet "
            "evening into something loud and alive."
        ),
    ]

    sorter = LLMSorter(api_key=api_key)
    result = sorter.sort(
        items=passages,
        prompt="Sort the list of literary passages by reading level, from lowest to highest.")

    expected = [
        # Lowest reading level
        (
            "The cat sat on the mat. It saw a bug and jumped up. "
            "The bug flew away. The cat looked and then lay down again."
        ),
        # Middle reading level
        (
            "The wind moved through the trees and made the leaves whisper. "
            "Maya stood on the porch and watched the sky darken. "
            "She felt nervous but excited, knowing the storm would change the quiet "
            "evening into something loud and alive."
        ),
        # Highest reading level
        (
            "The wind’s passage through the trees became a kind of language, "
            "articulate in its insistence, as if the world were attempting to speak "
            "before being interrupted. Standing at the threshold, Maya sensed the "
            "approaching storm not merely as weather, but as a force of "
            "transformation, one that would fracture the evening’s fragile calm and "
            "reassemble it into something unrecognizable."
        ),
    ]
    assert result == expected
