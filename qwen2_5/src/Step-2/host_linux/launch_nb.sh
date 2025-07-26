if [ "$VIRTUAL_ENV" == "" ] && [ "$CONDA_DEFAULT_ENV" == "" ]; then
  echo "Failed to launch. Please activate a conda or python virtual environment before launching the notebook."
  exit 1;
fi
python3 -m pip install --upgrade pip
pip install -r ../../requirements.txt
pip install -r requirements.txt

ln -s ../assets assets

echo "Launching Jupyter Notebook"

jupyter notebook --ip=$HOSTNAME --no-browser --allow-root &

