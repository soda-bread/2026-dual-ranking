from src.offline_moo_adapter import (
    build_offline_moo_problem,
    is_offline_moo_problem,
)


def canonical_real_world_problem_name(problem_name):
    return None


def is_real_world_problem(problem_name):
    return is_offline_moo_problem(problem_name)


def build_real_world_problem(problem_name):
    if is_offline_moo_problem(problem_name):
        return build_offline_moo_problem(problem_name)

    raise ValueError(f"Unknown real-world problem: {problem_name}")


def get_real_world_problem_y_bounds(problem_name):
    return None, None
