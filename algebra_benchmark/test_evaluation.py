"""
Test evaluation functions for algebra benchmarks using real examples from REL-A1.
"""

from algebra_benchmark.evaluation import evaluate_response


def test_algebra_examples():
    """Test REL-A1 (Raven's Progressive Matrix) evaluation with 3 examples."""
    examples = [
        {
            "question": "Complete the Raven's progressive matrix. Only return the missing panel index (0-7)!\nrow 1: (639, 25, 275), (223, 736, 677), (892, 87, 421); row 2: (639, 25, 275), (223, 736, 677), (892, 87, 421); row 3: (639, 25, 275), (223, 736, 677), \n\nAnswer set:\nAnswer 0: (639, 25, 275)\nAnswer 1: (93, 233, 602)\n...",
            "answer": {"target": 0},
            "responses": [
                "Answer 0",  # Correct (label matches index)
                "0",  # Correct (0-based index)
                "Answer 1",  # Incorrect
            ]
        },
        {
            "question": "Complete the Raven's progressive matrix. Only return the missing panel index (0-7)!\nrow 1: (718, 213, 499), (885, 643, 143), (140, 745, 539); row 2: (718, 213, 499), (885, 643, 143), (140, 745, 539); row 3: (718, 213, 499), (885, 643, 143), \n\nAnswer set:\nAnswer 0: (898, 399, 219)\n...\nAnswer 6: (718, 213, 499)\nAnswer 7: (761, 766, 128)",
            "answer": {"target": 6},
            "responses": [
                "Answer 6",  # Correct (label matches index)
                "6",  # Correct (0-based index)
                "Answer 0",  # Incorrect
            ]
        },
        {
            "question": "Complete the Raven's progressive matrix. Only return the missing panel index (0-7)!\nrow 1: (539, 730, 201), (312, 995, 650), (438, 518, 121); row 2: (539, 730, 201), (312, 995, 650), (438, 518, 121); row 3: (539, 730, 201), (312, 995, 650), \n\nAnswer set:\nAnswer 0: (64, 21, 554)\n...\nAnswer 3: (539, 730, 201)\n...",
            "answer": {"target": 3},
            "responses": [
                "Answer 3",  # Correct (label matches index)
                "3",  # Correct (0-based index)
                "The answer is 3",  # Correct (extracted as 3)
            ]
        },
    ]
    
    print("=" * 80)
    print("Testing REL-A1 (Raven's Progressive Matrix) Evaluation")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold Answer: target index {example['answer']['target']} (Answer {example['answer']['target']})")
        
        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-A1")
            print(f"\n  Response {j}: {response}")
            print(f"    Predicted index: {result.get('pred')} (Answer {result.get('pred') if result.get('pred') is not None else None})")
            print(f"    Correct: {result.get('correct')}")
            assert result.get('gold') == example['answer']['target'], f"Gold mismatch: {result.get('gold')} != {example['answer']['target']}"


if __name__ == "__main__":
    test_algebra_examples()
    print("\n" + "=" * 80)
    print("All algebra evaluation tests completed!")
    print("=" * 80)
