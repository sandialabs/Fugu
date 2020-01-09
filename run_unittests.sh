#!/bin/sh
echo "Running tests using DS backend"
python -m unittest -f test.test_suites.ds_suite
echo '>>>---<<<'

echo "Running tests using SNN backend"
#python -m unittest -f test.test_suites.snn_suite
echo '>>>---<<<'

#echo "Running tests using PYNN.BRIAN backend"
#python -m unittest test.test_suites.pynn_brian_suite
#echo '>>>---<<<'
