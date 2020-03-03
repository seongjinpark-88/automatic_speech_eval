# Kaldi 기초

This repository is from https://github.com/michaelcapizzi/kaldi_instructional, which contains a simplified version of `Kaldi` toolkit used for instructional purposes.

원 Repository의 내용을 한국어로 바꾸었으며, 이해하는데 필요한 부분들을 추가하였습니다. 그리고 일정에 맞추어 4개의 작은 단원으로 나누었습니다. 

* Language Model
* MFCC and Acoustic Model
* Finite State Transducer and HCLG
* Decoding and Evaluation

* **Note:** 필요한 파일을 다운로드하는 과정이 포함되어 있기 때문에, 디스크에 15GB 이상의 여유 공간이 필요합니다. 

## Resources

Resource 폴더 아래에 코드에서 사용한 자료들이 모여있습니다. 

## Installation

이번에 진행되는 워크샵에서는 Docker를 사용하여 코드를 실행하는데 필요한 환경을 구축하게 됩니다. 윈도우 OS를 사용하시는 경우, 윈도우 환경에서 Ubuntu를 설치하신 후 Linux 버전의 Docker를 설치하는 것을 권장드리며, MacOS의 경우 Mac 버전의 Docker를 설치하시면 됩니다. 

* Windows 10 이상에서 Ubuntu 설치법: https://code.iamseapy.com/archives/108
* Linux에서 Docker 설치법: https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1 에서 `install docker ce` 부분을 참고하시면 됩니다. 
* Mac에서 Docker 설치법: https://docs.docker.com/docker-for-mac/install/
* 전반적인 Docker 설치법: https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html#linux

**Note:** Windows Ubuntu의 경우 https://blog.aliencube.org/ko/2018/04/11/running-docker-and-azure-cli-from-wsl/ 를 참고하여 Docker를 설치한 이후, Ubuntu를 실행하시면 됩니다. 그리고 다음의 명령어를 통해서 컴퓨터의 c 드라이브를 Ubuntu에서도 불러올 수 있도록 설정을 해야 합니다. 설정을 마친 이후에 Ubuntu를 종료 후 다시 시작하시면 됩니다. 이후 모든 작업은 `/c/Users/사용자이름/Desktop/scratch`에서 이루어집니다.

```bash
sudo echo '[automount]\nroot = /\noptions = "metadata"]' >> wsl.conf
sudo mv wsl.conf /etc/

## Ubuntu 재시작 후

cd /c/Users/사용자이름/Desktop/scratch
```

**Note:** Windows 10 Pro 이하의 버전에서는 Hyper-V 지원의 문제로 ubuntu 및 Docker for Windows가 설치되지 않습니다. [Windows 10 Home에서 Docker 설치하기](https://gwonsungjun.github.io/how%20to%20install/2018/01/28/Dockerinstall/)에서 내용을 확인하시고 설치하시면 되겠습니다. 바탕화면에 생성된 `Docker Quickstart Terminal`은 마우스 우클릭 후 관리자 권한으로 실행하여주셔야 합니다. 

**Note:** `Docker ToolBox`에서는 잘못된 사용을 방지하기 위해서 `Symbolic link(바로가기)`의 생성을 막아두었습니다. 혹 관련된 문제가 발생하는 경우 [이 곳](https://jessezhuang.github.io/article/virtualbox-tips/)에서 `How to Create Virtualenv in Shared Foler` 부분에 나오는 명령어를 실행하신 이후, `Docker Quickstart Terminal`을 관리자 권한으로 실행하여주시면 됩니다. 

**Note:** Windows 10 Pro 이하 버전에서 사용하는 `Docker Toolbox`는 수동으로 CPU 및 메모리 수를 변경하여야 합니다. 기본적으로 CPU는 1개, 메모리는 1GB로 설정되어 있기 때문에 `kaldi` 실행이 어렵습니다. `Docker Quickstart Terminal`을 관리자 권한으로 실행하신 이후, 아래 명령어를 이용하여 값을 변경하실 수 있습니다. CPU나 메모리의 값은 본인의 컴퓨터 사양을 확인하시고 변경하시는 것이 좋으며, 아래는 CPU 4개, 4GB의 Docker를 구성하는 명령어입니다. 

```bash
docker-machine stop
cd /c/Program\ Files/Oracle/VirtualBox
./VBoxManage modifyvm default --cpus 4
./vBoxManage modifyvm default --memory 4096
docker-machine start
```



### kaldi 내려받기

이번 워크샵에서 사용하는 kaldi에 해당하는 파일들은 github에서 내려받으실 수 있습니다. 다음과 같은 순서로 명령어를 실행하세요. # 이후에 작성된 내용은 다음 줄에 있는 코드에 대한 설명입니다. 

```bash
# 작업을 위한 scratch 폴더를 / 아레에 생성
sudo mkdir /scratch

# /scratch 폴더로 이동
cd /scratch

# git에서 kaldi_instructional 내려받기
sudo git clone https://github.com/seongjinpark-88/kaldi

# kaldi 폴더로 이동
cd kaldi

# git에 존재하는 branch 다운로드
sudo git fetch

# 모든 파일을 최신 상태로 업데이트
sudo git checkout
```

앞으로 진행될 `kaldi` 관련 내용은 모두 `/scratch/kaldi` 폴더 아래에서 진행됩니다.

* **Note:** Windows 10 Home 버전의 경우 바탕화면에서 작업을 진행합니다. 바탕화면에 `scratch` 폴더를 생성하신 이후 `Docker Quickstart Terminal`을 관리자 권한으로 실행, 아래 명령어를 통해 `scratch` 폴더로 이동하시면 됩니다. `sudo` 명령어가 사용이 불가하므로, `sudo`를 제외한 나머지 부분을 위와 동일하게 입력하여주시면 됩니다. 

```bash
cd ~/Desktop/scratch

```

### kaldi을 위한 `Docker` 설치

* 다음의 명령어를 사용하여서 kaldi 실행에 필요한 `Docker` 환경을 설치할 수 있습니다. **build_container.sh** 파일은 docker 설치에 필요한 `Dockerfile`의 내용을 불러와서 실제 docker container를 설치하는 과정을 수행하는 스크립트입니다. 

```bash
cd /scratch/kaldi/docker
sudo ./build_container.sh

```

* **Note:** Windows 10 Home 버전을 사용하시는 경우, 아래의 명령어를 사용하시면 됩니다. 

```bash
cd ~/Desktop/scratch/docker
./build_container.sh -w
```

* **Note:** Windows 10 Pro 이상을 사용하시는 경우, 아래의 명령어를 사용하시면 됩니다.

```bash
cd /c/Users/사용자이름/Desktop/scratch/docker
./build_container.sh -w
```


* **Note:** 원 저자는 저자의 github에 있는 kaldi_instructional에 있는 코드들을 실행하기 위한 Docker 환경을 DockerHub에 업로드 하였습니다. 다음과 같은 명령어를 통해서 `Docker hub`에서 해당 Docker container를 설치할 수 있습니다. 하지만 이 경우 원 저자의 github의 내용을 모두 가져온 이후 `jupyter notebook` 파일들을 삭제, 워크샵에서 사용하는 github 파일들을 붙여넣은 이후 사용하여야 합니다. 필요하신 분들을 위해 다음의 코드를 남깁니다. 아래 `bash` 명령어를 실행하신 이후, 워크샵에서 사용하는 `kaldi` github에서 **egs/INSTRUCTIONAL** 디렉토리를 /scratch/kaldi_instructional/egs 아래에 붙여넣으시면 됩니다.  

```bash
sudo mkdir /scratch/

cd /scratch/

sudo git clone https://github.com/michaelcapizzi/kaldi_instructional.git

cd kaldi_instructional

sudo git fetch

sudo git checkout kaldi_instructional

docker pull mcapizzi/kaldi_instructional

cd egs

sudo rm -rf INSTRUCTIONAL

```

* **Note:** 윈도우 10에서 실행한 우분투에서 Docker를 사용하는 방법은 https://blog.aliencube.org/ko/2018/04/11/running-docker-and-azure-cli-from-wsl/ 를 참고하세요. 


## `Docker` container 실행

`Docker` container를 구성한 상태에서는 프로젝트 폴더 (`/scratch/kaldi`) 에 있는 `start_container.sh` 파일을 실행시킴으로써 `Docker` container를 실행할 수 있습니다. `start_container.sh` 파일을 실행하게 되면 컴퓨터에서 `8880` 포트를 개방하게 되며, 해당 포트에서 이후 사용될 `Jupyter Notebook`을 실행하게 됩니다. 

```
seongjinpark@seongjin:~$ cd /scratch/kaldi/
seongjinpark@seongjin:/scratch/kaldi$ sudo ./start_container.sh 
[sudo] password for seongjinpark: 
root@b0b49e44f58b:/scratch/kaldi/egs/INSTRUCTIONAL# 
```

`Docker` container 내부에 들어가게 되면 사용자 계정이 `root`로 변경되고, 맨 마지막에 `$` 대신 `#`으로 끝나게 됩니다. 

## `Jupyter` 실행

`Jupyter Notebook`은 오픈소스 웹 어플리케이션으로, 웹 브라우저를 이용하여 코드를 작성하고 공유할 수 있게 해주는 툴입니다. `Docker` 내부에 들어와 있는 상태라면 `./start_jupyter.sh` 파일을 실행하여 `jupyter` 실행을 위한 URL을 받을 수 있습니다. 

아래는 `./start_jupyter.sh` 파일을 실행한 예입니다. 

```
root@b0b49e44f58b:/scratch/kaldi/egs/INSTRUCTIONAL# ./start_jupyter.sh 
[I 12:13:20.353 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 12:13:20.734 NotebookApp] Serving notebooks from local directory: /scratch/kaldi/egs/INSTRUCTIONAL
[I 12:13:20.734 NotebookApp] 0 active kernels
[I 12:13:20.734 NotebookApp] The Jupyter Notebook is running at:
[I 12:13:20.734 NotebookApp] http://0.0.0.0:8880/?token=5ba706c2b543f5a70b4226ab8990c68d7c508ec3d56a59c0
[I 12:13:20.735 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 12:13:20.735 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://0.0.0.0:8880/?token=5ba706c2b543f5a70b4226ab8990c68d7c508ec3d56a59c0
```

다음과 같은 화면이 나온다면, `http://0.0.0.0:8880/?token=5ba706c2b543f5a70b4226ab8990c68d7c508ec3d56a59c0` (0.0.0.0:8880 뒤에 내용은 사용자마다 다를 수 있습니다) 를 사용하시는 브라우저 (크롬이나 파이어폭스 사용을 권장합니다) 주소창에 붙여넣으시면 `Jupyter` 창을 확인하실 수 있습니다. `jupyter notebook`의 사용 방법은 [이 곳](https://dojang.io/mod/page/view.php?id=1157) 을 참고하세요. 

**Note:** 현재 `0.0.0.0` 대신 임의의 알파벳+숫자의 조합으로 ip 주소가 출력되는 에러가 있습니다. 그럴 경우 해당 링크를 복사하여 붙여넣으신 이후  `http://`와 `:8880` 사이의 글자들을 **`0.0.0.0`** 으로 변경하셔서 다시 브라우저에서 열어주시면 됩니다. 

**Note:** Windows 10 Home 버전 사용자의 경우 **0.0.0.0** 대신에 `Docker Quickstart Terminal` 실행 시 나오는 IP 주소를 입력하시면 됩니다. Windows 10 Pro 이상 사용자의 경우는 **0.0.0.0** 대신 **127.0.0.1**을 사용하시면 됩니다. 

## `tmux`를 이용하여 `jupyter` 실행하기

`jupyter`를 실행하게 되면, command-line 상에서는 다른 명령어를 실행할 수 없게 됩니다. 물론 다른 prompt 창을 열어서 실행할 수 있지만, `tmux`를 사용하면 편하게 `jupyter` 환경과 `Docker` 환경을 번갈아가며 사용할 수 있습니다. 

`./start_container.sh` 파일을 실행한 이후, `./start_jupyter.sh` 파일을 실행하기 전, 다음과 같은 명령어를 통해 `tmux` 창을 열 수 있습니다. 

```
# container 환경 실행
sudo ./start_container.sh

# tmux 실행
tmux new-session -s docker
```
`new-session` 명령어는 새로운 `tmux` 세션을 실행하겠다는 명령어이며, `-s` 는 새로운 세션의 이름을 정하겠다는 명령어입니다. `docker`는 새로 시작할 세션의 이름입니다. 

`tmux`를 실행하고 나면 맨 아래쪽에 녹색 표시줄이 생기게 됩니다. 

새로운 환경에서 이전 설명처럼 `jupyter`를 실행하신 후, `ctrl+b`를 누르신 이후 `d` 버튼을 누르시게 되면, `Docker` 환경으로 돌아와 다른 명령어를 입력할 수 있게 됩니다. 다시 `tmux` 세션으로 돌아가려면 아래 명령어를 입력하시면 됩니다. (`tmux` 세션 이름을 다르게 설정하셨을 경우, 아래 명령어에서 `docker` 대신 사용하신 세션의 이름을 입력하시면 됩니다.) 

`tmux attach -t docker`

그리고 실행하신 `tmux` 세션을 종료하실 때에는 아래 명령어를 입력하시면 됩니다. 

`tmux kill-session -t docker`
