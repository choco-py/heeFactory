# ubuntu 18.04 Installation
- cuda 10.0
- cudnn 7.4
- conda 4.7.11
- tensorflow 1.13.0

## generate env
<pre><code>
conda --version


conda create -n <가상환경명> python=3.7 tensorflow

conda remove --name <가상환경명> --all

conda info --envs

# 처음 한번만 source로 activate 해주기

source activate <가상환경명>
source deactivate <가상환경명>

# 다음부터는 이렇게!

conda activate <가상환경명>
conda deactivate <가상환경명>

</pre></code>



## jupyter notebook
<pre><code>
#step1. install notebook
pip3 install notebook

#step2. modify notebook config (remote)
vi home/users/.jupyter/jupyter_notebook_config.py

  c = get_config()
  c.NotebookApp.allow_origin='*'
  c.NotebookApp.notebook_dir=''
  c.NotebookApp.ip=''
  c.NotebookApp.passwd=u''
  c.NotebookApp.port=''
  c.NotebookApp.open_browser=False

#step3. 가상환경을 jupyter kernel로 추가해주기

source activate [가상환경명]
pip install ipykernel
python -m ipykernel install --user --name [가상환경명] --display-name '가상환경명커널명'

</pre></code>


## User management
<pre><code>

sudo adduser [USERNAME]
sudo deluser [USERNAME]
</pre></code>


현재 계정:
guest/guest
hsohee/09896
su/skccdata
datasicence-sh/ds04

## rdkit installation


<pre><code>
anaconda search -t conda boost
#적합한 boost library 찾아서 설치
#conda-forge/boost
sudo conda install --channel https://conda.anaconda.org/conda-forge boost

anaconda search -t conda rdkit
#acellera/rdkit
conda install --channel https://conda.anaconda.org/acellera rdkit

</pre></code>

### Error
EnvironmentNotWritableError: The current user does not have write permissions to the target environment.
conda 위치 권한 변경해주기

<pre><code>
sudo chown -R $USER:$USER /opt/conda
</pre></code>


