from problems import problem1, problem2, problem3
from searchMPP import SearcherMPP
from stripsForwardPlanner import Forward_STRIPS
from stripsHeuristic import maxh, h1, h2


def distance_heuristic(state):
    products = [obj for obj in state if obj.startswith('box')]
    total_distance = 0
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            total_distance += abs(i - j)
    return total_distance


def solve_problem_with_heuristic(problem, heuristic):
    print(f"Solving problem: {problem} with heuristic: {heuristic.__name__}")
    try:
        searcher = SearcherMPP(Forward_STRIPS(problem, heuristic))
        solution = searcher.search()
        if solution:
            print("Solution found:")
            solution = str(solution).split(",")
            for step, action in enumerate(solution):
                print(f"Step {step + 1}: {action}")
    except TimeoutError:
        print("No solution found within the timeout.")


def solve_problem(problem):
    solve_problem_with_heuristic(problem, h1)
    solve_problem_with_heuristic(problem, h2)
    solve_problem_with_heuristic(problem, maxh(h1, h2))
    print("\n\n")


def solve_all_problems():
    solve_problem(problem1)
    solve_problem(problem2)
    solve_problem(problem3)


if __name__ == "__main__":
    solve_all_problems()
