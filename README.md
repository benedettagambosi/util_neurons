# util_neurons
General purpose neurons

Tested with:
*branch main*: Ubuntu 18.04.5 LTS, Python 3.8.6 and NEST Release 2.18.0

### Installation instructions

0. Install NEST following the instructions provided here (http://www.nest-simulator.org/)
1. Set the following environment variables (either into '.bashrc' or using 'export' from the shell)
```
NEST_INSTALL_DIR= <nest-simulator-install-folder>
NEST_DATA_DIR=$NEST_INSTALL_DIR/share/nest
NEST_DOC_DIR=$NEST_INSTALL_DIR/share/doc/nest
NEST_MODULE_PATH=$NEST_INSTALL_DIR/lib/nest
NEST_PYTHON_PREFIX=$NEST_INSTALL_DIR/lib/python3.8/site-packages
SLI_PATH=$NEST_INSTALL_DIR/share/nest/sli
LD_LIBRARY_PATH=$NEST_INSTALL_DIR/lib/nest:$LD_LIBRARY_PATH
PATH=$NEST_INSTALL_DIR/bin:$PATH
PYTHONPATH=$NEST_PYTHON_PREFIX${PYTHONPATH:+:$PYTHONPATH}
```
2. Clone this GitHub Repository in a directory outside NEST source and build directories. E.g.:
```
cd $HOME
git clone https://github.com/cristianoalessandro/util_neurons.git
```
3. Move to target directory
```
cd $HOME/util_neurons/target
```
4. Recompile nest with the additional module "util_neurons"
```
cmake -Dwith-nest=$NEST_INSTALL_DIR/bin/nest-config .
make all
make install
```

#### Using the util_neurons module

You should be able to use the module into your NEST application by adding the following line of code in your Python script
```
nest.Install("util_neurons_module")
```

