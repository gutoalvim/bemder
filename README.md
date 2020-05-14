BEMDER
=======

This is an open-source framework for 3D Boundary Elements Method (BEM) acoustic simulations in python with the package Bempp-cl.

Installation
============

Windows
--------

pT-Br
--------

Tutorial Instalação Bempp - Obs. Todos os textos em CAPS são individuais para cada usuário. Comandos estão entre '$ $'(não copiar o $)

1 - Baixar e instalar o Anaconda Python 3.7 -- https://www.anaconda.com/distribution/#download-section

2 - Abrir Anaconda Prompt criar um novo ambiente -- $ conda create -n ENVNAME python=3.7 $ - (Obs. ENVNAME é o nome de seu ambiente)

3 - Ativar o novo ambiente no Anacoda Prompt com -- $ conda activate ENVNAME $

4 -- $ conda install git $ 

5 - Criar uma pasta para os pacotes de simulação -- $ mkdir %UserProfile%\Documents\PYTHON_BEM $

6 - Navegar até a pasta -- $ cd %UserProfile%\Documents\PYTHON_BEM $

7 -- $ git clone https://github.com/bempp/bempp-cl.git $

8 -- $ git clone https://github.com/gutoalvim/bemder.git $

9 -- $ cd bemder $

10 -- $ pip install pyopencl-2020.1+cl12-cp37-cp37m-win_amd64.whl $

11 - Instalar os pacotes necessários -- $ pip install numpy scipy meshio plotly numba matplotlib$

12 -- $conda install quaternion$

13 - Instalar o Gmsh no Anaconda -- $ conda install -c conda-forge gmsh $

14 - Instalar os ambientes de desenvolvimento preferidos. O Spyder é semelhante ao Matlab. O Jupyter conta com uma interface intuitiva e ferramentas de visualização 3D exclusivas dessa plataforma -- $ conda install spyder $ -- $ conda install jupyter $

Spyder
******
15 - No Spyder (digite $ spyder $ no Ananconda Prompt), acessar o Python Path Manager (ícone com logo do Python na barra superior), clicar [+ Add Path] e selecionar a pasta do bempp-cl, faça a mesma coisa com a pasta do bemder.

16 - Teste rodando o script Examples/default_test.py

Jupyter
******
16 - Para o jupyter, use o exemplo disponível para começar a simular.


Python interface
*****************
UNDER DEVELOPMENT
*****************
