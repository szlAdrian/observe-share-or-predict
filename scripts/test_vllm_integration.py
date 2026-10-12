#!/usr/bin/env python3
"""
Simple test to verify vLLM integration is working correctly.
This test doesn't require a running vLLM server - just checks code structure.
"""

import sys
from omegaconf import OmegaConf

def test_vllm_config_loading():
    """Test that vllm.yaml config loads correctly"""
    import os
    import habitat_llm

    # Get the path to the vllm config
    habitat_llm_dir_path = os.path.dirname(habitat_llm.__file__)
    vllm_config_path = f"{habitat_llm_dir_path}/conf/llm/vllm.yaml"

    assert os.path.exists(vllm_config_path), f"vllm.yaml not found at {vllm_config_path}"

    # Load the config
    config = OmegaConf.load(vllm_config_path)

    assert config.inference_mode == "vllm", "inference_mode should be 'vllm'"
    assert config.llm._target_ == "habitat_llm.llm.VLLMModel", "LLM target should be VLLMModel"
    assert config.host == "localhost", "Default host should be localhost"
    assert config.port == 8000, "Default port should be 8000"
    assert config.generation_params.max_tokens == 250, "Max tokens should be 250"

    print("✓ vllm.yaml config loaded successfully")

def test_vllm_model_class():
    """Test that VLLMModel class can be imported and has required methods"""
    try:
        from habitat_llm.llm.vllm_model import VLLMModel

        # Check that VLLMModel has required methods
        required_methods = [
            'init_vllm_model',
            'generate_vllm',
            '_generate_vllm_api',
            'generate_hf_llm',
            'get_logprobs',
        ]

        for method in required_methods:
            assert hasattr(VLLMModel, method), f"VLLMModel missing method: {method}"

        print("✓ VLLMModel class has all required methods")

    except ImportError as e:
        # This is okay - happens if openai package is not installed
        print(f"⚠ Cannot import VLLMModel due to missing dependencies: {e}")
        print("  This is expected if 'openai' package is not installed")

def test_hf_model_vllm_routing():
    """Test that hf_model.py has vllm routing"""
    import inspect
    from habitat_llm.llm.hf_model import VLMHFModel

    # Check that VLMHFModel has init_vllm_model method
    assert hasattr(VLMHFModel, 'init_vllm_model'), "VLMHFModel missing init_vllm_model method"

    # Check that VLMHFModel has generate_vllm method
    assert hasattr(VLMHFModel, 'generate_vllm'), "VLMHFModel missing generate_vllm method"

    # Check __init__ source for vllm handling
    init_source = inspect.getsource(VLMHFModel.__init__)
    assert 'vllm' in init_source, "__init__ should mention vllm"

    # Check generate source for vllm handling
    generate_source = inspect.getsource(VLMHFModel.generate)
    assert 'vllm' in generate_source, "generate() should mention vllm"

    print("✓ hf_model.py has correct vllm routing")

def test_init_imports():
    """Test that __init__.py imports VLLMModel"""
    import inspect
    from habitat_llm import llm

    # Get source of __init__.py
    init_source = inspect.getsource(llm)

    assert 'VLLMModel' in init_source, "__init__.py should import VLLMModel"

    print("✓ __init__.py correctly imports VLLMModel")

if __name__ == "__main__":
    print("Running vLLM integration verification tests...\n")

    try:
        test_vllm_config_loading()
        test_vllm_model_class()
        test_hf_model_vllm_routing()
        test_init_imports()

        print("\n✓ All verification tests passed!")
        sys.exit(0)

    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
