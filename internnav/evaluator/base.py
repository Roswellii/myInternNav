from internnav.configs.evaluator import EvalCfg
from internnav.env import Env
from internnav.utils.comm_utils.client import AgentClient


class Evaluator:
    """
    Base class of all evaluators.
    """

    evaluators = {}

    def __init__(self, config: EvalCfg):
        self.config = config
        self.env = Env.init(config.env, config.task)
        self.agent = AgentClient(config.agent)

    def eval(self):
        raise NotImplementedError

    @classmethod
    def register(cls, evaluator_type: str):
        """
        Register a evaluator class.
        """

        def decorator(evaluator_class):
            if evaluator_type in cls.evaluators:
                raise ValueError(f"Evaluator {evaluator_type} already registered.")
            cls.evaluators[evaluator_type] = evaluator_class

        return decorator

    @classmethod
    def init(cls, config: EvalCfg):
        """
        Init a evaluator instance from a config.
        """
        if config.eval_type not in cls.evaluators:
            error_msg = f"Evaluator type '{config.eval_type}' is not registered. Available evaluators: {list(cls.evaluators.keys())}"
            if config.eval_type.startswith('habitat'):
                error_msg += (
                    f"\n\nHabitat evaluators require additional dependencies. "
                    f"If you're trying to use '{config.eval_type}', please ensure:\n"
                    f"1. Habitat-sim and habitat-lab are installed\n"
                    f"2. depth_camera_filtering is installed: pip install git+https://github.com/naokiyokoyama/depth_camera_filtering.git\n"
                    f"3. All habitat extensions are properly imported"
                )
            raise KeyError(error_msg)
        return cls.evaluators[config.eval_type](config)
