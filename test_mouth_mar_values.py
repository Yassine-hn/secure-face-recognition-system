#!/usr/bin/env python3
"""Test script to verify MAR values with different mouth landmark configurations."""

import numpy as np
from liveness.checks.mouth import mouth_aspect_ratio

# Test with the new mouth indices: [61, 84, 17, 314, 405, 320, 307, 375]
def test_mar_values():
    print("Testing MAR values with different mouth configurations:")
    print("=" * 50)
    
    # Test 1: Perfectly closed mouth (straight line)
    closed_mouth = [
        (50, 100),  # left corner
        (70, 100),  # left upper
        (90, 100),  # upper center
        (110, 100), # upper right
        (130, 100), # right corner
        (110, 100), # lower right
        (90, 100),  # lower center
        (70, 100),  # lower left
    ]
    mar = mouth_aspect_ratio(closed_mouth)
    print(f"Perfectly closed mouth: MAR = {mar:.4f}")
    
    # Test 2: Natural closed mouth (slight curve)
    natural_closed = [
        (50, 100),  # left corner
        (70, 98),   # left upper
        (90, 97),   # upper center
        (110, 97),  # upper right
        (130, 100), # right corner
        (110, 102), # lower right
        (90, 103),  # lower center
        (70, 102),  # lower left
    ]
    mar = mouth_aspect_ratio(natural_closed)
    print(f"Natural closed mouth: MAR = {mar:.4f}")
    
    # Test 3: Slightly open mouth
    slightly_open = [
        (50, 100),  # left corner
        (70, 95),   # left upper
        (90, 92),   # upper center
        (110, 92),  # upper right
        (130, 100), # right corner
        (110, 108), # lower right
        (90, 111),  # lower center
        (70, 108),  # lower left
    ]
    mar = mouth_aspect_ratio(slightly_open)
    print(f"Slightly open mouth: MAR = {mar:.4f}")
    
    # Test 4: Clearly open mouth
    open_mouth = [
        (50, 100),  # left corner
        (70, 85),   # left upper
        (90, 80),   # upper center
        (110, 80),  # upper right
        (130, 100), # right corner
        (110, 120), # lower right
        (90, 125),  # lower center
        (70, 120),  # lower left
    ]
    mar = mouth_aspect_ratio(open_mouth)
    print(f"Clearly open mouth: MAR = {mar:.4f}")
    
    # Test 5: Very wide open mouth
    very_open = [
        (50, 100),  # left corner
        (70, 70),   # left upper
        (90, 60),   # upper center
        (110, 60),  # upper right
        (130, 100), # right corner
        (110, 140), # lower right
        (90, 150),  # lower center
        (70, 140),  # lower left
    ]
    mar = mouth_aspect_ratio(very_open)
    print(f"Very wide open mouth: MAR = {mar:.4f}")
    
    print("\n" + "=" * 50)
    print("Threshold Analysis:")
    print(f"Absolute threshold 0.4:")
    print(f"  Natural closed passes: {mouth_aspect_ratio(natural_closed) >= 0.4}")
    print(f"  Slightly open passes: {mouth_aspect_ratio(slightly_open) >= 0.4}")
    print(f"  Open mouth passes: {mouth_aspect_ratio(open_mouth) >= 0.4}")
    
    print(f"\nRelative increase threshold 0.2:")
    baseline = mouth_aspect_ratio(natural_closed)
    print(f"  Baseline (natural closed): {baseline:.4f}")
    print(f"  Slightly open increase: {mouth_aspect_ratio(slightly_open) - baseline:.4f} (passes: {(mouth_aspect_ratio(slightly_open) - baseline) >= 0.2})")
    print(f"  Open mouth increase: {mouth_aspect_ratio(open_mouth) - baseline:.4f} (passes: {(mouth_aspect_ratio(open_mouth) - baseline) >= 0.2})")

if __name__ == "__main__":
    test_mar_values()
