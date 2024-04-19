#!/bin/bash

set -o noglob

TESTPYPI="--extra-index-url https://test.pypi.org/simple/ --pre"

MODULE="pyincore_incubator"
CONDA_IMPORTS="scip>=8.0.0"

# setup pip-compile to resolve dependencies, also is minimal version of python
if [ ! -d /tmp/pyincore-requirements ]; then
  python3 -m virtualenv /tmp/pyincore-requirements
  . /tmp/pyincore-requirements/bin/activate
  pip install pip-tools
else
  . /tmp/pyincore-requirements/bin/activate
fi

# all requirements in pyincore_viz files
rm -f requirements.imports
for x in ${MODULE} tests notebooks; do
  case $x in
    notebooks)
      FILES="*.ipynb"
      FOLDER="${MODULE}"
      ;;
    *)
      FILES="*.py"
      FOLDER="${x}"
      ;;
  esac
  IMPORTS=$(egrep -R -h --include "${FILES}" '(import|from) ' ${FOLDER} | \
            sed -e 's/^ *"//' -e 's/\\n",$//' | \
            egrep '^(from|import)' | \
            awk "!/${MODULE}/ { print \$2 }" | \
            sort -u | \
            egrep -v '^(\(|_)' | \
            sed 's/,/ /g')
  if [ "${IMPORTS}" == "" ]; then continue; fi
  # check which imports are not standard python
  rm -f requirements.tmp
  for i in $IMPORTS; do
    MISSING=$(python3 -c "import $i" 2>&1 | \
      awk '/ModuleNotFoundError/ { print $5 }' | \
      sed -e 's/yaml/pyyaml/' -e 's/jose/python-jose/' -e 's/_pytest/pytest/' -e 's/PIL/pillow/' -e 's/osgeo/gdal/' -e "s/'//g")
    if [ "$MISSING" == "gdal" ]; then
      echo "'$MISSING' needs to be installed"
    elif ! grep "${MISSING}" requirements.imports &>/dev/null ; then
      echo ${MISSING} >> requirements.tmp
    fi
  done
  sort -u requirements.tmp > requirements.${x}
  cat requirements.${x} >> requirements.imports
done

# find latest version of all packages needed to install
rm -f requirements.in
for p in $(cat requirements.imports ); do
  PACKAGE="$p"
  VERSION=$(grep "^${PACKAGE}" 2>/dev/null requirements.min | sed "s/^${PACKAGE}//")
  echo "$p$VERSION" >> requirements.in
done
pip-compile ${TESTPYPI} --quiet --upgrade --rebuild --output-file requirements.tmp requirements.in
cat requirements.tmp | grep -v ' *#.*' | grep -v '^$' | grep -v "^--" > requirements.ver

# create a markdown file with what we found.
cat << EOF > versions.md
# ${MODULE} dependencies version list

This file is autogenerated and lists the latest versions of all dependencies. The table will list package (pip and conda), minimum version and latest version found.

| PIP Package | Conda Package | Required Version | Matched Version (pip) |
|-------------|---------------|------------------|-----------------------|
EOF

# create the environment.yml file for conda. This is intended to setup a conda environment for
# development on pyincore_viz.
cat << EOF > environment.yml
name: ${MODULE}
channels:
  - conda-forge
  - defaults
dependencies:
EOF
for r in ${CONDA_IMPORTS}; do
  echo "  - $r" >> environment.yml
done

# create the requirements.txt file for pip
cat << EOF > requirements.txt
# This file is autogenerated
EOF

# all dependencies found
for p in $(cat requirements.imports | sort -u ); do
  if [ "$p" == "" ]; then continue; fi
  PACKAGE="$p"
  LATEST=$(grep "^${PACKAGE}" 2>/dev/null requirements.ver | sed "s/^${PACKAGE}//")
  VERSION=$(grep "^${PACKAGE}" 2>/dev/null requirements.min | sed -e "s/^${PACKAGE}//")
  MD_VERSION=$(echo $VERSION | sed -e 's/\([<>]\)/\\\1/g')
  if [ "${VERSION}" == "" ]; then
    echo "NO VERSION SPECIFIED FOR $p"
  fi
  LATEST=$(grep "^${PACKAGE}==" 2>/dev/null requirements.ver | sed "s/^${PACKAGE}==//")
  CONDA=$(echo ${PACKAGE} | sed -e 's/fastjsonschema/python-fastjsonschema/' \
                        -e 's/ipython-genutils/ipython_genutils/' \
                        -e 's/jupyter-core/jupyter_core/' \
                        -e 's/jupyter-client/jupyter_client/' \
                        -e 's/jupyterlab-pygments/jupyterlab_pygments/' \
                        -e 's/jupyterlab-widgets/jupyterlab_widgets/' \
                        -e 's/prometheus-client/prometheus_client/' \
                        -e 's/stack-data/stack_data/')
  echo -n "| ${PACKAGE} " >> versions.md
  if [ "${CONDA}" == "${PACKAGE}" ]; then
    echo -n "| " >> versions.md
  else
    echo -n "| ${CONDA} " >> versions.md
  fi
  echo -n "| ${MD_VERSION}" >> versions.md
  echo -n "| ${LATEST} " >> versions.md
  echo "|" >> versions.md

  echo "  - ${p}${VERSION}" >> environment.yml
  echo "${p}${VERSION}" >> requirements.txt
  sed -i~ -e "s/    - ${p}$/    - ${p}${VERSION}/" -e "s/    - ${p}>.*/    - ${p}${VERSION}/" recipes/meta.yaml
  sed -i~ "s/^\(  *\)'$p.*'/\1'${p}${VERSION}'/" setup.py
done

# document all other dependencies
for r in $(cat requirements.ver); do
  PACKAGE=${r%%=*}
  if grep "$PACKAGE" requirements.imports 2>&1 >/dev/null; then continue; fi
  VERSION=$(grep "^${PACKAGE}" 2>/dev/null requirements.min | sed "s/^${PACKAGE}//")
  CONDA=$(echo ${PACKAGE} | sed -e 's/fastjsonschema/python-fastjsonschema/' \
                        -e 's/ipython-genutils/ipython_genutils/' \
                        -e 's/jupyter-core/jupyter_core/' \
                        -e 's/jupyter-client/jupyter_client/' \
                        -e 's/jupyterlab-pygments/jupyterlab_pygments/' \
                        -e 's/jupyterlab-widgets/jupyterlab_widgets/' \
                        -e 's/prometheus-client/prometheus_client/' \
                        -e 's/stack-data/stack_data/')
  echo -n "| ${PACKAGE} " >> versions.md
  if [ "${CONDA}" == "${PACKAGE}" ]; then
    echo -n "| " >> versions.md
  else
    echo -n "| ${CONDA} " >> versions.md
  fi
  if [ "$VERSION" != "" ]; then
    echo -n "| ~~${VERSION}~~" >> versions.md
  else
    echo -n "| " >> versions.md
  fi
  echo -n "| ${r##*==} " >> versions.md
  echo "|" >> versions.md
done

## cleanup
rm -f requirements.${MODULE} requirements.tests requirements.notebooks requirements.tmp requirements.ver  requirements.in *~
