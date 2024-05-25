# create virtual environment
echo "Creating virtual environment"
python -m venv ds4b_course_env
# activate virtual environment
echo "Activate the virtual environment, upgrade pip and install the packages into the virtual environment"
source ds4b_course_env/bin/activate && pip install --upgrade pip && pip install -r requirements.txt