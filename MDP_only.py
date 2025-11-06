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


if __name__ == '__main__':
  my_mdp = RacecarMDP()

