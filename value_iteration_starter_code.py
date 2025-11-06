# -*- coding: utf-8 -*-

""" 
# ----------------------------------------------- # 
# artificial intelligence                         #
# csci 3385/5385/6385                             #
# fall 2025                                       #
# value iteration                                 #
# ----------------------------------------------- #

The following are what we know from the outset:
1) States: {Cool, Warm, Overheated}
2) Actions: {Slow, Fast}
3) Transition probabilities:
	  P(Cool | Cool, Slow) 	= 1.0
	  P(Cool | Cool, Fast) = 0.5
	  P(Warm | Cool, Fast) = 0.5
	  P(Cool | Warm, Slow) = 0.5
	  P(Warm | Warm, Slow) = 0.5
	  P(Warm | Warm, Fast) = 0.9
	  P(Overheated | Warm, Fast) = 0.1
4) Reward function:
	  R(Cool, Slow, Cool) = +10
	  R(Cool, Fast, Cool) = +15
	  R(Cool, Fast, Warm) = +15
	  R(Warm, Slow, Cool) = +10
	  R(Warm, Slow, Warm) = +10
	  R(Warm, Fast, Warm) = +20
	  R(Warm, Fast, Overheated) = -50

From these we want to compute solutions to the Bellman equation for both
the optimal policy and the expected utility of being in a state under the
optimal policy:

solution = {'Cool':
             {'action': <'Slow' or 'Fast'>,
              'utility': <#>},
            'Warm':
             {'action': <'Slow' or 'Fast'>,
              'utility': <#>}}

Once we have filled out the entries in the solution, we can use it as follows:
- solution['Cool']['action'] : Optimal policy action to take in the Cool state.
- solution['Warm']['action'] : Optimal policy action to take in the Warm state.
- solution['Cool']['utility']: Expected utility of being in the Cool state and acting optimally.
- solution['Warm']['utility']: Expected utility of being in the Warm state and acting optimally.
"""

import copy

class RacecarMDP:
  """
  # This class defines the racecar MDP.
  """

  def __init__(self):

    # states
    self.states = ["Cool", "Warm", "Overheated"]

    # terminal_states
    self.is_terminal = {"Cool": False,
                        "Warm": False,
                        "Overheated": True}

    # start state
    self.current_state = "Cool"

    # actions
    self.actions = ["Slow", "Fast"]

    # Transition model P(s' | s, a)
    # Example: transition_model[('Cool', 'Slow')]['Overheated'] == P(over | cool, slow)
    self.transition_model = {
      ("Cool", "Slow"): {"Cool": 1.0, "Warm": 0.0, "Overheated": 0.0}, # P(s' | Cool, Slow)
      ("Cool", "Fast"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0}, # P(s' | Cool, Fast)
      ("Warm", "Slow"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0}, # P(s' | Warm, Slow)
      ("Warm", "Fast"): {"Cool": 0.0, "Warm": 0.9, "Overheated": 0.1}  # P(s' | Warm, Fast)
    }


  def reward(self, state, action, next_state):

    if next_state == 'Overheated':
      return -50

    elif state == 'Warm' and action == 'Fast':
      return 20

    elif state == 'Cool' and action == 'Fast':
      return 15

    elif state == 'Warm' and action == 'Slow':
      return 10

    elif state == 'Cool' and action == 'Slow':
      return 10


class Agent:

  def __init__(self):

    # This is what we want to fill in with the correct values using the
    # value iteration algorithm.
    self.solution = {
      'Cool': {'action': '0', 'utility': 0},
      'Warm': {'action': '0', 'utility': 0}
    }

  def print_policy(self):

    print('state  |  action    utility')
    print('-------|-------------------')
    print('Cool   |  ' + self.solution['Cool']['action'] + '      ' + str(round(self.solution['Cool']['utility'], 4)))
    print('Warm   |  ' + self.solution['Warm']['action'] + '      ' + str(round(self.solution['Warm']['utility'], 4)))


  def calculate_bellman_solutions(self, U, mdp, gamma):
    # Once we have accurate utility estimates, we can compute solutions for
    # the optimal policy.

    for state in mdp.states:
      if mdp.is_terminal[state]:
        continue

      max_q_value = -999
      max_q_action = '0'

      for action in mdp.actions:
        q = self.Q(state, action, gamma, mdp, U)

        if q > max_q_value:
          max_q_value = q
          max_q_action = action

      self.solution[state]['action'] = max_q_action
      self.solution[state]['utility'] = max_q_value


  def Q(self, state, action, gamma, mdp, U):
    """
    # Fill this function in for the next assignment.
    """
    q = 0

    for next_state, prob in mdp.transition_model[(state, action)].items():

      # -------------------------------------- #
      # Your code here                         #
      # -------------------------------------- #

    return q


  def value_iteration(self, mdp, gamma, epsilon=1e-7, iter_limit=100000):
    """
    # This function implements the value iteration algorithm.
    """
    assert 0 <= gamma < 1, 'gamma must be in [0, 1) for convergence to work in this MDP.'

    # Initialize utility function with U[state] = 0 for all states.
    U_current = {state: 0.0 for state in mdp.states}


    i = 0  # for tracking how many iterations have been made

    # The while-loop will keep running until converged becomes True, or i
    # becomes greater than iter_limit.
    converged = False
    while not converged:

      # We will use delta to keep up with the maximum change in a state's utility estimate
      delta = 0

      i += 1

      U_old = copy.deepcopy(U_current)

      for state in mdp.states:

        # If this is a terminal state, we can skip it
        if mdp.is_terminal[state]:
          continue

        # -------------------------------------- #
        # Your code here                         #
        # -------------------------------------- #


        # Calculate |U_current - U_old| and update delta if needed.
        utility_change = abs(U_current[state] - U_old[state])
        if utility_change > delta:
          delta = utility_change

      # Check if convergence threshold is met:
      if delta <= epsilon:
        converged = True
        print('Utilities converged on iteration ' + str(i))
        print()

      if i > iter_limit and not converged:
        print('state utility estimates did not converge within iteration limit.')
        break


    # From the converged utilities, calculate optimal policy.
    # policy = {'Cool': 'optimal_action', 'Warm': 'optimal_action'}
    # return policy
    if converged:
      self.calculate_bellman_solutions(U_current, mdp, gamma)


if __name__ == '__main__':
  my_mdp = RacecarMDP()
  my_agent = Agent()
  my_agent.value_iteration(my_mdp, gamma=0.9)
  my_agent.print_policy()

