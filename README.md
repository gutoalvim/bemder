BEMDER
=======

This is an open-source framework for 3D Boundary Elements Method (BEM) acoustic simulations in python with the package Bempp-cl.

Installation
============

Windows
--------

pT-Br
--------

Tutorial Instalação Bempp - Obs. Todos os textos em CAPS são individuais para cada usuário

1 - Baixar e instalar o Anaconda Python 3.7 -- https://www.anaconda.com/distribution/#download-section
2 - Abrir Anaconda Prompt criar um novo ambiente -- conda create env -n ENVNAME python=3.7 (Obs. ENVNAME é o nome de seu ambiente)
3 - Ativar o novo ambiente no Anacoda Prompt com -- conda actiavte ENVNAME
4 -- Conda install git 
5 - Criar uma pastar para os pacotes de simulação -- mkdir C:\Users\USER\Documents\PYTHON_BEM
6 - Navegar até a pasta -- cd C:\Users\USER\Documents\PYTHON_BEM
7 -- git clone https://github.com/bempp/bempp-cl.git
8 -- git clone https://github.com/gutoalvim/bemder.git
9 -- pip install pyopencl-2019.1.2+cl12-cp37-cp37m-win_amd64.whl
10 - Instalar os pacotes necessários -- pip install numpy scipy meshio plotly numba
11 - Instalar o Gmsh no Anaconda -- conda install -c conda-forge gmsh
12 - Instalar os ambientes de desenvolvimento preferidos. O Spyder é semelhante ao Matlab. O Jupyter conta com uma interface intuitiva e ferramentas de visualização 3D exclusivas dessa plataforma -- conda install spyder -- conda install jupyter

Spyder
******
13 - No Spyder, acessar o Python Path Manager (ícone com logo do Python na barra superior), clicar [+ Add Path] e selecionar a pasta do bempp-cl, faça a mesma coisa com a pasta do bemder.

Jupyter
******
14 - Para o jupyter, use o exemplo disponível para começar a simular.


Python interface
*****************
UNDER DEVELOPMENT
*****************
