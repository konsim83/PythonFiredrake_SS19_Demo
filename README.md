# Example Project with Firedrake

This Python code is used for teaching purposes and is a modification of
the 
[Helmholtz tutorial](https://www.firedrakeproject.org/demos/helmholtz.py.html)
of Firedrake.

*To run it you will need:*
	
* A **Linux** distribution (I recommend Ubuntu 20.04 for beginners)
* **Python3** and **curl** v3.6 or greater (try to use the package repositories to install them)
* **[Paraview](www.paraview.org)** for the visualization (free of charge)
* **[Firedrake](https://www.firedrakeproject.org/)**

---
**HINT**

Follow Firedrake's [installation instructions](https://www.firedrakeproject.org/download.html). It will install Firedrake 
in the folder where you run it from and build it in a virtual environment so it will not mess with your system. The installer may ask you to install missing packages.

*Note*: On Windows Subsystem for Linux I advise you to use the --minimal-petsc option so that the installation command should look like

```
python3 firedrake-install --minimal-petsc
```
After that you will need to activate the environment (just follow the instructions). 

You can also use the [Eclipse IDE](https://www.eclipse.org/) (free of charge) together with the PyDev package that you can get
in the Eclipse Marketplace (Menu bar --> Help --> Eclipse Marketplace). In eclipse you can than import the code through `Projects from folder or archive`. Then edit the project settings and change the project natures to `Python nature`. Or use any Python IDE that you prefer. 

*Note* that at any rate you will need to point the IDE to use the `python3` executable that the Firedrake installation build for you. 

Firedrake has a great [documentation](https://www.firedrakeproject.org/documentation.html).

---	

## Getting and running the examples

You must first clone the repository:

```
git clone https://github.com/konsim83/Demo_Firedrake.git
```
and activate firedrake's virtual environment

```
source your-firedrake-install-folder/bin/activate
```

Then enter the code folder

```
cd Demo_Firedrake/
```
and type for example

```
python3 src/helmholtz_1d.py
```
Open the `vtk` files that were produced in the `data/` folder with Paraview to see the result.
