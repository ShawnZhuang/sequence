RUN_HOME=$(realpath $(dirname ${BASH_SOURCE[0]})/)

source ${RUN_HOME}/env.source
echo ${PYTHON_PATH}

# python3 tests/python/action/test_action.py 
python3  tests/python/block/test_attention.py