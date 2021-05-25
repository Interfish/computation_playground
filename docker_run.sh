SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker run -it --gpus=all --rm -v ${SCRIPT_DIR}:/root/computation_playground computation_playground:latest