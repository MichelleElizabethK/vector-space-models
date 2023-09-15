import argparse
from vector_space_model import VectorSpaceModel

# Expected usage:
# ./run -q topics.xml -d documents.lst -r run -o sample.res ...

# Where:
#   -q topics.xml -- a file including topics in the TREC format 
#   -d documents.lst -- a file including document filenames 
#   -r run -- a string identifying the experiment (will be inserted in the
#      result file as "run_id")
#   -o sample.res -- an output file 


def main():
    parser = argparse.ArgumentParser(description="Command Line Arguments for Vector Space Model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-q", help="a file including topics in the TREC format")
    parser.add_argument("-d", help="a file including document filenames")
    parser.add_argument(
        "-r", help="a string identifying the experiment (will be inserted in the result file as \"run_id\"")
    parser.add_argument("-o", help="an output file")

    args = parser.parse_args()
    config = vars(args)
    vector_space_model = VectorSpaceModel()
    vector_space_model.process_documents(config['q'], config['d'], config['r'])
    vector_space_model.create_queries()
    vector_space_model.get_output(config['o'], config['r'])


if __name__ == '__main__':
    main()
