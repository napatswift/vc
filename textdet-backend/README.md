# Label Studio ML Backend

This directory contains a script for the Label Studio ML backend. To build the backend, we need to use the `label-studio-ml` package. The only requirement is that the script must contain a subclass of `LabelStudioMLBase`. In this directory, the script is called `model.py`.

To build the project, run the following command:

```
label-studio-ml init --script ./model.py mlback
```

Once the project directory is created, run the following command to serve it:

```
label-studio-ml start ./mlback
```