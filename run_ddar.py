# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run DD+AR or AlphaGeometry solver.

Please refer to README.md for detailed instructions.
"""

import traceback

from absl import app
from absl import flags
from absl import logging
import ddar
import graph as gh
#import lm_inference as lm
import pretty as pt
import problem as pr


_GIN_SEARCH_PATHS = flags.DEFINE_list(
    'gin_search_paths',
    ['third_party/py/meliad/transformer/configs'],
    'List of paths where the Gin config files are located.',
)
_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', ['base_htrans.gin'], 'List of Gin config files.'
)
_GIN_PARAM = flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.'
)

_PROBLEMS_FILE = flags.DEFINE_string(
    'problems_file',
    'imo_ag_30.txt',
    'text file contains the problem strings. See imo_ag_30.txt for example.',
)
_PROBLEM_NAME = flags.DEFINE_string(
    'problem_name',
    'imo_2000_p1',
    'name of the problem to solve, must be in the problem_file.',
)
_MODE = flags.DEFINE_string(
    'mode', 'ddar', 'either `ddar` (DD+AR) or `alphageometry`')
_DEFS_FILE = flags.DEFINE_string(
    'defs_file',
    'defs.txt',
    'definitions of available constructions to state a problem.',
)
_RULES_FILE = flags.DEFINE_string(
    'rules_file', 'rules.txt', 'list of deduction rules used by DD.'
)
_CKPT_PATH = flags.DEFINE_string('ckpt_path', '', 'checkpoint of the LM model.')
_VOCAB_PATH = flags.DEFINE_string(
    'vocab_path', '', 'path to the LM vocab file.'
)
_OUT_FILE = flags.DEFINE_string(
    'out_file', '', 'path to the solution output file.'
)  # pylint: disable=line-too-long
_BEAM_SIZE = flags.DEFINE_integer(
    'beam_size', 1, 'beam size of the proof search.'
)  # pylint: disable=line-too-long
_SEARCH_DEPTH = flags.DEFINE_integer(
    'search_depth', 1, 'search depth of the proof search.'
)  # pylint: disable=line-too-long

DEFINITIONS = None  # contains definitions of construction actions
RULES = None  # contains rules of deductions


def natural_language_statement(logical_statement: pr.Dependency) -> str:
  """Convert logical_statement to natural language.

  Args:
    logical_statement: pr.Dependency with .name and .args

  Returns:
    a string of (pseudo) natural language of the predicate for human reader.
  """
  names = [a.name.upper() for a in logical_statement.args]
  names = [(n[0] + '_' + n[1:]) if len(n) > 1 else n for n in names]
  return pt.pretty_nl(logical_statement.name, names)


def proof_step_string(
    proof_step: pr.Dependency, refs: dict[tuple[str, ...], int], last_step: bool
) -> str:
  """Translate proof to natural language.

  Args:
    proof_step: pr.Dependency with .name and .args
    refs: dict(hash: int) to keep track of derived predicates
    last_step: boolean to keep track whether this is the last step.

  Returns:
    a string of (pseudo) natural language of the proof step for human reader.
  """
  premises, [conclusion] = proof_step

  premises_nl = ' & '.join(
      [
          natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
          for p in premises
      ]
  )

  if not premises:
    premises_nl = 'similarly'

  refs[conclusion.hashed()] = len(refs)

  conclusion_nl = natural_language_statement(conclusion)
  if not last_step:
    conclusion_nl += ' [{:02}]'.format(refs[conclusion.hashed()])

  #return f'{premises_nl} \u21d2 {conclusion_nl}'
  return f'{premises_nl} -: {conclusion_nl}'


def write_solution(g: gh.Graph, p: pr.Problem, out_file: str) -> None:
  """Output the solution to out_file.

  Args:
    g: gh.Graph object, containing the proof state.
    p: pr.Problem object, containing the theorem.
    out_file: file to write to, empty string to skip writing to file.
  """
  setup, aux, proof_steps, refs = ddar.get_proof_steps(
      g, p.goal, merge_trivials=False
  )

  solution = '\n=========================='
  solution += '\n * From theorem premises:\n'
  premises_nl = []
  for premises, [points] in setup:
    solution += ' '.join([p.name.upper() for p in points]) + ' '
    if not premises:
      continue
    premises_nl += [
        natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
        for p in premises
    ]
  solution += ': Points\n' + '\n'.join(premises_nl)

  solution += '\n\n * Auxiliary Constructions:\n'
  aux_premises_nl = []
  for premises, [points] in aux:
    solution += ' '.join([p.name.upper() for p in points]) + ' '
    aux_premises_nl += [
        natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
        for p in premises
    ]
  solution += ': Points\n' + '\n'.join(aux_premises_nl)

  # some special case where the deduction rule has a well known name.
  r2name = {
      'r32': '(SSS)',
      'r33': '(SAS)',
      'r34': '(Similar Triangles)',
      'r35': '(Similar Triangles)',
      'r36': '(ASA)',
      'r37': '(ASA)',
      'r38': '(Similar Triangles)',
      'r39': '(Similar Triangles)',
      'r40': '(Congruent Triangles)',
      'a00': '(Distance chase)',
      'a01': '(Ratio chase)',
      'a02': '(Angle chase)',
  }

  solution += '\n\n * Proof steps:\n'
  for i, step in enumerate(proof_steps):
    _, [con] = step
    nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
    rule_name = r2name.get(con.rule_name, '')
    #nl = nl.replace('\u21d2', f'{rule_name}\u21d2 ')
    nl = nl.replace('-:', f'{rule_name}-:')
    solution += '{:03}. '.format(i + 1) + nl + '\n'

  solution += '==========================\n'
  logging.info(solution)
  if out_file:
    with open(out_file, 'w') as f:
      f.write(solution)
    logging.info('Solution written to %s.', out_file)


def run_ddar(g: gh.Graph, p: pr.Problem, out_file: str) -> bool:
  """Run DD+AR.

  Args:
    g: gh.Graph object, containing the proof state.
    p: pr.Problem object, containing the problem statement.
    out_file: path to output file if solution is found.

  Returns:
    Boolean, whether DD+AR finishes successfully.
  """
  ddar.solve(g, RULES, p, max_level=1000)

  goal_args = g.names2nodes(p.goal.args)
  if not g.check(p.goal.name, goal_args):
    logging.info('DD+AR failed to solve the problem.')
    return False

  write_solution(g, p, out_file)
  return True


def main(_):
  global DEFINITIONS
  global RULES

  # definitions of terms used in our domain-specific language.
  DEFINITIONS = pr.Definition.from_txt_file(_DEFS_FILE.value, to_dict=True)
  # load inference rules used in DD.
  RULES = pr.Theorem.from_txt_file(_RULES_FILE.value, to_dict=True)

  # when using the language model,
  # point names will be renamed to alphabetical a, b, c, d, e, ...
  # instead of staying with their original names,
  # in order to match the synthetic training data generation.
  need_rename = _MODE.value != 'ddar'

  # load problems from the problems_file,
  problems = pr.Problem.from_txt_file(
      _PROBLEMS_FILE.value, to_dict=True, translate=need_rename
  )

  if _PROBLEM_NAME.value not in problems:
    raise ValueError(
        f'Problem name `{_PROBLEM_NAME.value}` '
        + f'not found in `{_PROBLEMS_FILE.value}`'
    )

  this_problem = problems[_PROBLEM_NAME.value]

  if _MODE.value == 'ddar':
    g, _ = gh.Graph.build_problem(this_problem, DEFINITIONS)
    run_ddar(g, this_problem, _OUT_FILE.value)

  else:
    raise ValueError(f'Unknown FLAGS.mode: {_MODE.value}')


if __name__ == '__main__':
  app.run(main)
