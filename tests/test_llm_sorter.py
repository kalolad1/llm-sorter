"""Tests for the LLMSorter class.

This module contains test cases that verify the LLMSorter's ability to
sort various types of data using LLM-based comparisons, including integers,
strings, and complex text passages.
"""

import random

import pytest

from src.llm_sorter import LLMSorter


@pytest.mark.parametrize("input_list,expected", [
    ([], []),
    ([5], [5]),
    ([3, 1], [1, 3]),
    ([3, 1, 4, 1, 5, 9, 2, 6], [1, 1, 2, 3, 4, 5, 6, 9]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([2, 2, 2, 1, 1], [1, 1, 2, 2, 2]),
])
def test_sort_integers(api_key: str, input_list: list[int], expected: list[int]) -> None:
    """Test sorting of integer lists with various edge cases.

    Verifies that the LLMSorter correctly sorts integers in ascending order,
    handling empty lists, single elements, already sorted lists, and duplicates.

    Args:
        api_key: The OpenRouter API key fixture.
        input_list: The unsorted list of integers to sort.
        expected: The expected sorted result.
    """
    sorter = LLMSorter(api_key=api_key)
    assert sorter.sort(items=input_list) == expected


@pytest.mark.parametrize("input_list,expected", [
    ([], []),
    (["a", "b", "c"], ["a", "b", "c"]),
    (["c", "b", "a"], ["a", "b", "c"]),
    (["a", "a", "b", "b", "c", "c"], ["a", "a", "b", "b", "c", "c"]),
])
def test_sort_strings(api_key: str, input_list: list[str], expected: list[str]) -> None:
    """Test sorting of string lists in alphabetical order.

    Verifies that the LLMSorter correctly sorts strings alphabetically,
    including handling of empty lists, already sorted lists, and duplicates.

    Args:
        api_key: The OpenRouter API key fixture.
        input_list: The unsorted list of strings to sort.
        expected: The expected alphabetically sorted result.
    """
    sorter = LLMSorter(api_key=api_key)
    result = sorter.sort(
        items=input_list,
        prompt="Sort the list of strings in ascending, alphebetical order.")
    assert result == expected


def test_sort_literary_passages_by_complexity(api_key: str) -> None:
    """Test sorting literary passages by reading level complexity.

    Verifies that the LLMSorter can semantically sort text passages
    based on their reading complexity, from simplest to most complex.

    Args:
        api_key: The OpenRouter API key fixture.
    """
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


def test_sort_support_tickets_by_urgency(api_key: str) -> None:
    """Test sorting support tickets by urgency level.

    Verifies that the LLMSorter can semantically sort customer support
    tickets based on urgency, considering severity, scope of impact,
    and security implications.

    Args:
        api_key: The OpenRouter API key fixture.
    """
    tickets = [
        # Highest urgency
        (
            "URGENT: Production is down for all users. The API returns 500 errors "
            "and customers cannot access the platform."
        ),
        # Lowest urgency
        (
            "Feature request: It would be nice if the dashboard had dark mode."
        ),
        # Medium urgency
        (
            "I can't reset my password. The reset email never arrives. "
            "I've tried multiple times and checked spam."
        ),
        # High urgency
        (
            "Possible security issue: I received a password reset email that I did not request. "
            "Please investigate."
        ),
    ]

    sorter = LLMSorter(api_key=api_key)
    result = sorter.sort(
        items=tickets,
        prompt=(
            "Sort the customer support tickets by urgency from highest to lowest. "
            "Consider severity, scope of impact, and security implications."
        )
    )

    expected = [
        (
            "URGENT: Production is down for all users. The API returns 500 errors "
            "and customers cannot access the platform."
        ),
        (
            "Possible security issue: I received a password reset email that I did not request. "
            "Please investigate."
        ),
        (
            "I can't reset my password. The reset email never arrives. "
            "I've tried multiple times and checked spam."
        ),
        (
            "Feature request: It would be nice if the dashboard had dark mode."
        ),
    ]

    assert result == expected


def test_sort_large_list_passages_by_reading_level(api_key: str) -> None:
    """Test sorting a large list of passages by reading level.

    Verifies that the LLMSorter can handle sorting 99 text passages
    (33 each of low, medium, and high reading levels) correctly,
    testing performance with larger datasets.

    Args:
        api_key: The OpenRouter API key fixture.
    """
    low_passage = "The dog ran. It was fast. It saw food and ate it."
    mid_passage = (
        "The dog ran through the yard and barked at the birds. "
        "It seemed happy, but it was also hungry."
    )
    high_passage = (
        "The dog's restless motion suggested an inward turbulence, "
        "as if its hunger were less a physical sensation and more a metaphor for longing."
    )
    low_passages = [low_passage] * 33
    mid_passages = [mid_passage] * 33
    high_passages = [high_passage] * 33
    passages = low_passages + mid_passages + high_passages
    random.shuffle(passages)
    sorter = LLMSorter(api_key=api_key)

    result = sorter.sort(
        items=passages,
        prompt="Sort the passages by reading level, from lowest to highest."
    )

    expected = low_passages + mid_passages + high_passages
    assert result == expected
