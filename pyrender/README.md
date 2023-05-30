# 1-point-per-pixel neural rendering comparison



# INSTALL NOTES:
## WINDOWS
1. ensure sdl2.dll exists. and set the PYSDL2_DLL_PATH variable in your environment to its folder.
2. make sure the cl.exe is in path.

# TO USE
1. Use the utility scripts to create a scene from a COLMAP reconstruction. (process described in [../README.md](/../README.md))
2. run main.py with '--make_workspace' and '-s C://Path/To/Your/Scene/'. Other initialization parameters are for the most part optional, but use '--batch_size=1' to train faster stochastically.
3. always specify a '--workspace C://Path/To/Your/Workspace/' to specify what workspace to use. Note that 'C://Path/To/Your' would need to exist and be a folder for this example, otherwise workspace creation would fail.
4. To continue using a workspace after its creation, do not specify '--make_workspace'. Any creation-related parameters will be mostly ignored, and the command can otherwise remain the same.
