set -e
PYTHON_VERSION=cp37-cp37m
yum -y remove cmake
/opt/python/${PYTHON_VERSION}/bin/pip install cmake twine
export PATH=$PATH:/opt/python/${PYTHON_VERSION}/bin/
/opt/python/${PYTHON_VERSION}/bin/pip wheel .
auditwheel repair hyperjet-*-${PYTHON_VERSION}-*.whl