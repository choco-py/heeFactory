## generate env
<pre><code>
conda --version


conda create -n <가상환경명> python=3.7 tensorflow

conda remove --name <가상환경명> --all

conda info --envs

source activate <가상환경명>
source deactivate <가상환경명>
</pre></code>

conda activate <가상환경명>
conda deactivate <가상환경명>

## jupyter notebook

pip3 install notebook

<pre><code>
vi home/users/.jupyter/jupyter_notebook_config.py

c = get_config()
c.NotebookApp.allow_origin='*'
c.NotebookApp.notebook_dir=''
c.NotebookApp.ip=''
c.NotebookApp.passwd=u''
c.NotebookApp.port=''
c.NotebookApp.open_browser=False
</pre></code>


<pre><code>
source activate [가상환경명]
pip install ipykernel
python -m ipykernel install --user --name [가상환경명] --display-name '가상환경명커널명'


</pre></code>


## User management

sudo adduser [USERNAME]
sudo deluser [USERNAME]


현재 계정:
guest/guest
hsohee/09896
su/skccdata
datasicence-sh/ds04

