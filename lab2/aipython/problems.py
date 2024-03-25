from stripsProblem import create_blocks_world, Planning_problem, clear, on

# Problem 1
blocks_dom1 = create_blocks_world({'a', 'b', 'c', 'd'})
initial_state1 = {
    on('a'): 'table',
    clear('a'): True,
    on('b'): 'c',
    clear('b'): True,
    on('c'): 'd',
    clear('c'): False,
    on('d'): 'b',
    clear('d'): False,
}
goal_state1 = {
    on('c'): 'a',
    on('b'): 'c',
    on('a'): 'table',
    on('d'): 'b'
}
problem1 = Planning_problem(blocks_dom1, initial_state1, goal_state1)

# Problem 2
blocks_dom2 = create_blocks_world({'a', 'b', 'c', 'd', 'e'})
initial_state2 = {
    on('a'): 'd',
    clear('a'): False,
    on('b'): 'a',
    clear('b'): True,
    on('c'): 'table',
    clear('c'): True,
    on('d'): 'b',
    clear('d'): False,
    on('e'): 'a',
    clear('e'): False
}
goal_state2 = {
    on('b'): 'a',
    on('e'): 'a',
    on('c'): 'b',
    on('a'): 'd',
}
problem2 = Planning_problem(blocks_dom2, initial_state2, goal_state2)

# Problem 3
blocks_dom3 = create_blocks_world({'a', 'b', 'c', 'd', 'e', 'f'})
initial_state3 = {
    on('a'): 'd',
    clear('a'): False,
    on('b'): 'e',
    clear('b'): True,
    on('c'): 'a',
    clear('c'): False,
    on('d'): 'table',
    clear('d'): False,
    on('e'): 'table',
    clear('e'): True,
    on('f'): 'a',
    clear('f'): True,
}
goal_state3 = {
    on('e'): 'table',
    on('c'): 'a',
    on('b'): 'e',
    on('d'): 'table',
    on('f'): 'd',
}
problem3 = Planning_problem(blocks_dom3, initial_state3, goal_state3)