# Deployment Information

For use on a new machine, we use conda. On the original machine, we always work
inside an isolated python environment, managed by conda. 

## Create

Environment made with conda. To make an environment;

`conda create --name <whatever> python=3.6`

## Activate
To go inside an environment, execute;

`conda activate <whatever>` or
`source activate <whatever>`

Then, your terminal gets a prefix, like so;

`$(whatever) user@machine:`

indicating that you are now _inside_ of this environment.

## Export
This environment can be exported to a `.yml` file through the following command:

`conda env export > environment.yml`

Which creates the `.yml` file present in the root dir. Important: you need to execute this command _inside_ the environment, how else would conda know what environment you would be talking about?

## Load
To recreate this environment, it suffices to run;

`conda env create -f environment.yml -n <whatever name you want>`

Which presupposes that you have an anaconda install running on your own machine.
In theory, this should be portable enough.

## Manually add to list in Jupyter

Sometimes it works by installing jupyter in the environment itself. But sometimes it does not, for unclear reasons.

However, the clean way is the following;

`source activate myenv`
`python -m ipykernel install --user --name myenv --display-name "Py(myenv)"`
