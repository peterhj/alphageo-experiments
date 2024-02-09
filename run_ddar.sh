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

# !/bin/bash
set -e
set -x

#virtualenv -p python3 .
#source ./bin/activate

#pip install --require-hashes -r requirements.txt

#gdown --folder https://bit.ly/alphageometry
#DATA=ag_ckpt_vocab

#MELIAD_PATH=meliad_lib/meliad
#mkdir -p $MELIAD_PATH
#git clone https://github.com/google-research/meliad $MELIAD_PATH
#export PYTHONPATH=$PYTHONPATH:$MELIAD_PATH

DDAR_ARGS=(
  --defs_file=$(pwd)/defs.txt \
  --rules_file=$(pwd)/rules.txt \
);

BATCH_SIZE=0
BEAM_SIZE=0
DEPTH=100

SEARCH_ARGS=(
  --beam_size=$BEAM_SIZE
  --search_depth=$DEPTH
)

#LM_ARGS=(
#  --ckpt_path=$DATA \
#  --vocab_path=$DATA/geometry.757.model \
#  --gin_search_paths=$MELIAD_PATH/transformer/configs \
#  --gin_file=base_htrans.gin \
#  --gin_file=size/medium_150M.gin \
#  --gin_file=options/positions_t5.gin \
#  --gin_file=options/lr_cosine_decay.gin \
#  --gin_file=options/seq_1024_nocache.gin \
#  --gin_file=geometry_150M_generate.gin \
#  --gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True \
#  --gin_param=TransformerTaskConfig.batch_size=$BATCH_SIZE \
#  --gin_param=TransformerTaskConfig.sequence_length=128 \
#  --gin_param=Trainer.restore_state_variables=False
#);

#echo $PYTHONPATH

# NB: ddar alone solves the following:

# --problems_file=$(pwd)/examples.txt
# --problem_name=orthocenter_aux

# --problems_file=$(pwd)/jgex_ag_231.txt
# --problem_name=examples/complete2/012/complete_002_6_GDD_FULL_41-60_59.gex

# --problems_file=$(pwd)/imo_ag_30.txt
# --problem_name=translated_imo_2000_p1 (after bugfix)
# --problem_name=translated_imo_2002_p2a
# --problem_name=translated_imo_2002_p2b
# --problem_name=translated_imo_2003_p4
# --problem_name=translated_imo_2004_p5 (fast)
# --problem_name=translated_imo_2005_p5 (after bugfix)
# --problem_name=translated_imo_2007_p4
# --problem_name=translated_imo_2010_p4
# --problem_name=translated_imo_2012_p1
# --problem_name=translated_imo_2013_p4
# --problem_name=translated_imo_2015_p4
# --problem_name=translated_imo_2016_p1
# --problem_name=translated_imo_2017_p4
# --problem_name=translated_imo_2022_p4

# NB: notable ddar fails:

# --problem_name=translated_imo_2008_p6 (slow)
# --problem_name=translated_imo_2009_p2 (fast)
# --problem_name=translated_imo_2011_p6 (high depth; timeout?)
# --problem_name=translated_imo_2012_p5 (fast)
# --problem_name=translated_imo_2014_p4 (fast, high depth)

# NB: errors ("Type Error: cannot unpack non-iterable Point object"):
# --problem_name=translated_imo_2000_p1
# --problem_name=translated_imo_2005_p5
# --problem_name=translated_imo_2008_p1a
# --problem_name=translated_imo_2008_p1b
# --problem_name=translated_imo_2019_p2

python -m run_ddar \
--alsologtostderr \
--problems_file=$(pwd)/reduced_imo_2009_p2.txt \
--problem_name=reduced_imo_2009_p2_v3 \
--mode=ddar \
"${DDAR_ARGS[@]}" \
"${SEARCH_ARGS[@]}"
#"${LM_ARGS[@]}"
