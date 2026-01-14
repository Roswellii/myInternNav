from internnav.habitat_extensions.habitat_env import HabitatEnv

try:
    from internnav.habitat_extensions.habitat_vln_evaluator import HabitatVLNEvaluator
except ImportError as e:
    print(f"Warning: Failed to import HabitatVLNEvaluator: {e}")
    HabitatVLNEvaluator = None
