import numpy as np
from dynamic_weights import DynamicWeights

def compare_outputs(original_output, refactored_output, test_name):
    is_equal = np.allclose(original_output, refactored_output, rtol=1e-10)
    print(f"\nTest: {test_name}")
    print(f"Original output: {original_output}")
    print(f"Refactored output: {refactored_output}")
    print(f"Outputs match: {is_equal}")
    return is_equal

def run_comparison_tests():
    initial_weights = [1.0, 2.0, 3.0]
    
    dw_original = DynamicWeights(weights=initial_weights.copy())
    dw_refactored = DynamicWeights(weights=initial_weights.copy())
    
    all_tests_passed = True
    
    # Test 1: Single Update
    print("\n=== Test 1: Single Update ===")
    test_cases = [
        (0, 1.0, 1),    # Positive reward for first domain
        (1, -0.5, 2),   # Negative reward for second domain
        (2, 2.0, 3),    # Large positive reward for third domain
    ]
    
    for index, reward, iteration in test_cases:
        orig_output = dw_original.update(index, reward, iteration)
        refac_output = dw_original.update(index, reward, iteration)
        
        test_passed = compare_outputs(
            orig_output, 
            refac_output,
            f"Single update (index={index}, reward={reward}, iteration={iteration})"
        )
        all_tests_passed &= test_passed

    # Test 2: Group Update
    print("\n=== Test 2: Group Update ===")
    group_test_cases = [
        ([0, 1], [1.0, -0.5], 1),
        ([1, 2], [-1.0, 2.0], 2),
        ([0, 1, 2], [0.5, 1.0, -0.5], 3),
    ]
    
    # Reset instances for group tests
    dw_original = DynamicWeights(weights=initial_weights.copy())
    dw_refactored = DynamicWeights(weights=initial_weights.copy())
    
    for indices, rewards, iteration in group_test_cases:
        orig_output = dw_original.group_update(indices, rewards, iteration)
        refac_output = dw_refactored.group_update(indices, rewards, iteration)
        
        test_passed = compare_outputs(
            orig_output,
            refac_output,
            f"Group update (indices={indices}, rewards={rewards}, iteration={iteration})"
        )
        all_tests_passed &= test_passed

    # Final Results
    print("\n=== Final Results ===")
    print(f"All tests {'passed' if all_tests_passed else 'failed'}")

if __name__ == "__main__":
    run_comparison_tests()