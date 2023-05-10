#INSTALL NOTES:
1) ensure sdl2.dll exists. as in the PYSDL2_DLL_PATH variable
2) make sure cl.exe is in path.

#TO USE
1)use the utility scripts to create a scene
2)run main with '--make_workspace' and '-s C://Path/To/Your/Scene/'. Other initialization parameters are optional.
3)always specify a '--workspace C://Path/To/Your/Workspace/' to specify what workspace to use. Note that 'C://Path/To/Your' would need to exist and be a folder for this example, otherwise workspace creation would fail.
4)To continue using a workspace after its creation, do not specify '--make_workspace'. Any creation-related parameters will be mostly ignored.
